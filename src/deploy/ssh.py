import asyncio
from dataclasses import dataclass
import io
import logging
import os
import re
import subprocess
import sys
import tarfile
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import aiofiles
import asyncssh
from asyncssh import SSHClientConnection

from src.deploy.deploy_utils import invalidate, make_header_text
from src.paths import Pathlike


asyncssh.set_debug_level(3)
asyncssh.set_log_level(logging.NOTSET)


@dataclass
class RunResult:
    exit_status: int
    stdout: str
    stderr: str
    exception: Optional[Exception] = None


class NoTermSSHClientConnection(asyncssh.SSHClientConnection):
    def _process_pty_req(self, packet):
        return False


class LogBuffer:
    def __init__(self):
        self.text = ""
        self.handlers = []
        self.ansi_escape = re.compile(r"\x1b\[[0-9;]*m")

    def clear(self):
        self.text = ""

    def _process_text(self, text, level, color_func=None):
        # text = self.ansi_escape.sub('', text)
        self.text += text + "\n"
        processed_text = color_func(text) if color_func else text
        for handler in self.handlers:
            handler(level, processed_text)
        return text

    def info(self, text):
        self._process_text(text, "info")

    def warning(self, text):
        self._process_text(text, "warning", lambda t: f"[yellow]{t}[/yellow]")

    def error(self, text):
        self._process_text(text, "error", lambda t: f"[red]{t}[/red]")


class SSH:
    def __init__(self, remote):
        self.remote = remote
        self.connection: Optional[SSHClientConnection] = None
        self.sftp = None
        self.ip = None
        self.port = None
        self.username = None
        self.logs = LogBuffer()
        self._log = logging.getLogger("DiscoSSH")

        # Constants
        self.max_size = 1_000_000_000  # 1GB

        # Configuration
        self.config = {
            "enable_urls": True,
            "enable_print_upload": False,
            "enable_print_download": True,
            "print_rsync": True,
        }

        # URLs for potential downloads
        self.urls = {
            "sd-v1-5.ckpt": "",
            "vae.vae.pt": "",
        }

    @property
    def log(self):
        self._log.name = "DiscSSH"
        if self.remote is not None and self.remote.instance is not None:
            self._log = logging.getLogger(f"{self.remote.instance.id}")
        return self._log

    async def is_connected(self):
        """Check if SSH connection is established and still active."""
        if self.connection is None:
            return False
        try:
            # Send a keepalive packet to verify connection is still responsive
            await self.connection.run("true")
            return True
        except (asyncssh.Error, ConnectionError, OSError):
            # Connection is broken or closed
            self.connection = None
            self.sftp = None
            return False

    def reset(self):
        self.logs.clear()
        if self.connection:
            self.connection.close()
            self.connection = None
            self.sftp = None

    def debug(self, msg):
        self.log.debug(msg)

    def info(self, msg):
        self.log.info(msg)
        self.logs.info(msg)
        invalidate()

    def warning(self, msg):
        self.log.warning(msg)
        self.logs.warning(msg)
        invalidate()

    def error(self, msg):
        self.log.error(msg)
        self.logs.error(msg)
        invalidate()

    async def connect(self, hostname, port, username, password=None, key_filename=None):
        """Establish an SSH connection."""
        if self.connection is not None:
            self.error("Attempting to connect when already connected.")
            return

        self.ip = hostname
        self.port = port
        self.username = username

        self.info(
            f"Connecting to {hostname}:{port} as {username} (password: {password}, key: {key_filename})"
        )
        self.connection = await asyncssh.connect(
            hostname,
            port=port,
            username=username,
            # password=password,
            # client_keys=key_filename,
        )
        self.connection.set_keepalive(30, 3)
        self.sftp = await self.connection.start_sftp_client()
        self.info(make_header_text(f"Connected to {username}@{hostname}:{port}"))

    async def disconnect(self):
        """Close the SSH connection."""
        if self.connection is None:
            self.error("Attempting to disconnect when not connected.")
            return

        self.connection.close()
        self.connection = None
        self.sftp = None
        self.info(make_header_text("Disconnected"))

    async def run_safe(self, cmd, cwd=None, *, log_all=False, stdout_handler=None):
        """Execute a command on the remote instance with improved output handling for both stdout and stderr."""
        try:
            return await self.run(
                cmd, cwd, log_all=log_all, stdout_handler=stdout_handler
            )
        except asyncssh.ProcessError as e:
            self.error(f"Error running command: {e}")
            return e.exit_status

    async def _run(self, *args, **kwargs):
        return await self.run(*args, **kwargs, log_cmd=False)

    async def run_stdout(self, *kargs, **kwargs) -> str:
        """
        Run a command and return the stdout.
        """
        result = await self.run(*kargs, **kwargs)
        return result.stdout.strip()

    async def run_exitcode(self, *kargs, **kwargs) -> int:
        """
        Run a command and return the exit code.
        """
        result = await self.run(*kargs, **kwargs)
        return result.exit_status

    async def run(
        self, cmd, cwd=None, *, log_all=False, log_cmd=True, stdout_handler=None
    ) -> RunResult:
        """
        Execute a command on the remote instance with improved output handling for both stdout and stderr.

        Major steps:
        1. Prepare the command
        2. Execute the command and handle output
        3. Process results
        """
        assert self.connection

        # 1. Prepare the command
        cmd = cmd.replace("'", '"')
        if cwd:
            cmd = f"cd {cwd} && {cmd}"

        if log_cmd or log_all:
            self.info(f"> {cmd}")
        # else:
        #     # semi-silently to logging lib
        #     self.log.info(f"> {cmd}")

        # 2. Execute the command and handle output
        try:
            process = await self.connection.create_process(cmd)

            async def process_output(stream, prefix, is_error):
                output = []
                async for line in stream:
                    line = line.rstrip("\n")
                    full_line = f"{prefix}: {line}"
                    full_line = (
                        f"[red]{full_line}[/red]" if is_error else f"[gray]{full_line}[/gray]"
                    )
                    if log_all:
                        self.info(full_line)
                    else:
                        self.log.debug(full_line)
                    if stdout_handler:
                        stdout_handler(full_line)
                    output.append(line)
                return "\n".join(output)

            # Process both streams concurrently
            stdout_output, stderr_output = await asyncio.gather(
                process_output(process.stdout, "STDOUT", False),
                process_output(process.stderr, "STDERR", True),
            )

            await process.wait()
            result = RunResult(
                process.exit_status if process.exit_status is not None else -1,
                stdout_output,
                stderr_output,
            )

        except asyncssh.ProcessError as e:
            self.error(f"Error running command: {e}")
            self.error(f"The command was: {cmd}")
            return RunResult(-1, "", "", e)

        except asyncssh.misc.ChannelOpenError as e:
            self.error(f"Channel open error: {e}")
            self.error(f"The command was: {cmd}")
            self.error("Attempting to reconnect ...")

            await self.connect(self.ip, self.port, self.username)

            return await self.run(cmd, cwd, log_all=log_all, log_cmd=log_cmd, stdout_handler=stdout_handler)

        return result

    async def run_detached(self, cmd, cwd=None, output_file=None):
        """
        Execute a command on the remote instance in a detached state,
        allowing it to continue running after disconnection.
        """
        assert self.connection

        cmd = cmd.replace("'", '"')
        if cwd is not None:
            cmd = f"cd {cwd} && {cmd}"

        if output_file is None:
            output_file = "/dev/null"

        # Use nohup to run the command in the background, immune to hangups
        detached_cmd = f"nohup {cmd} > {output_file} 2>&1 &"

        # Use 'disown' to remove the process from the shell's job control
        full_cmd = f"/bin/bash -c '{detached_cmd}; disown'"

        self.info(f"Starting detached process: {full_cmd}")

        result = await self.connection.run(full_cmd)

        if result.exit_status == 0:
            self.info("Detached process started successfully")
        else:
            self.log.error(
                f"Failed to start detached process. Exit status: {result.exit_status}"
            )
            self.log.error(f"Error output: {result.stderr}")

        return result.exit_status

    async def check_process(self, process_name):
        """Check if a process is running on the remote instance."""
        assert self.connection
        cmd = f"pgrep -f {process_name}"
        result = await self.connection.run(cmd)
        return result.exit_status == 0

    async def kill_process(self, process_name):
        """Kill a process on the remote instance."""
        cmd = f"pkill -f {process_name}"
        result = await self.run(cmd)
        return result.exit_status == 0

    async def clone_repo(self, repo: str, target_dir: Path):
        recursive = "--recursive" if "discore" in repo or "ComfyUI" in repo else ""
        cmd = f"git clone {recursive} {repo} {target_dir.as_posix()}"
        await self.run(cmd, log_all=False)

    async def read_file(self, path) -> str:
        """Read a text file from the remote instance and return it."""
        assert self.connection

        if isinstance(path, Path):
            path = path.as_posix()

        result = await self.connection.run(f"cat {path}")
        if result.exit_status == 0:
            return str(result.stdout).strip()
        else:
            self.error(f"Failed to read file {path}: {result.stderr}")
            return ""

    async def file_exists(self, path):
        """Check if a file exists on the remote instance."""
        if isinstance(path, Path):
            path = path.as_posix()

        result = await self._run(f"test -e {path}")
        return result.exit_status == 0

    async def is_mounted(self, local_path):
        if sys.platform.startswith("win"):
            self.info("Windows not supported yet.")
            return False
        elif sys.platform.startswith("darwin"):
            result = await asyncio.to_thread(
                subprocess.run,
                f'mount | grep -q "{local_path}"',
                shell=True,
                check=False,
            )
            return result.returncode == 0
        else:
            result = await asyncio.to_thread(
                subprocess.run,
                f"mountpoint -q {local_path}",
                shell=True,
                check=False,
            )
            return result.returncode == 0

    async def mount(self, local_path, remote_path, ip, port):
        """Mount a remote directory locally"""
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        if isinstance(remote_path, Path):
            remote_path = remote_path.as_posix()

        if sys.platform.startswith("win"):
            self.info("Windows not supported yet.")
            return
        elif sys.platform.startswith("darwin"):
            not_mounted = await asyncio.to_thread(
                subprocess.run,
                f'mount | grep -q "{local_path}"',
                shell=True,
                check=False,
            )
            not_mounted = not_mounted.returncode != 0
            if not_mounted:
                mount_cmd = f"sshfs root@{ip}:{remote_path} {local_path} -p {port} -o volname=Discore"
        else:
            not_mounted = await asyncio.to_thread(
                subprocess.run, f"mountpoint -q {local_path}", shell=True, check=False
            )
            not_mounted = not_mounted.returncode != 0
            if not_mounted:
                mount_cmd = f"sshfs root@{ip}:{remote_path} {local_path} -p {port}"

        if not_mounted:
            result = await asyncio.to_thread(
                subprocess.run, mount_cmd, shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                self.info("Mounted successfully!")
            else:
                self.info(f"Failed to mount. Error: {result.stderr}")
        else:
            self.info("Already mounted.")

    async def unmount(self, local_path):
        """Unmount a local directory"""
        local_path = Path(local_path)

        if sys.platform.startswith("win"):
            self.info("Windows not supported yet.")
            return
        elif sys.platform.startswith("darwin"):
            unmount_cmd = f"umount {local_path}"
        else:
            unmount_cmd = f"fusermount -u {local_path}"

        result = await asyncio.to_thread(
            subprocess.run, unmount_cmd, shell=True, capture_output=True, text=True
        )

        if result.returncode == 0:
            self.info("Unmounted successfully!")
        else:
            self.info(f"Failed to unmount. Error: {result.stderr}")

    # # TODO
    # local_path = None
    # if sys.platform.startswith('win'):
    #     self.info("Windows not supported yet.")
    #     return False
    # elif sys.platform.startswith('darwin'):
    #     return await asyncio.to_thread(subprocess.run, f'mount | grep -q "{local_path}"', shell=True, check=False).returncode == 0
    # else:
    #     return await asyncio.to_thread(subprocess.run, f'mountpoint -q {local_path}', shell=True, check=False).returncode == 0

    async def is_process_running(self, process_name):
        res = await self.run_stdout(
            f"ps aux | grep {process_name} | grep -v grep", log_cmd=False
        )
        return res != "" and res is not None

    async def put_any(
        self,
        source: Pathlike,
        target: Pathlike,
        forbid_rclone=False,
        force_rclone=False,
        forbid_recursive=False,
        rsync_excludes=None,
        rclone_includes=None,
        force=False,
    ):
        if rsync_excludes is None:
            rsync_excludes = []
        if rclone_includes is None:
            rclone_includes = []

        source = Path(source)
        target = Path(target)

        if not source.exists():
            self.error(f"File not found will be skipped: {str(source)}")
            return

        if force_rclone or source.is_dir():
            if force_rclone or not forbid_rclone:
                if sys.platform.startswith("linux"):
                    await self.put_rsync(
                        source,
                        target,
                        forbid_recursive,
                        rsync_excludes,
                        rclone_includes,
                    )
                else:
                    await self.put_rclone(
                        source,
                        target,
                        forbid_recursive,
                        rsync_excludes,
                        rclone_includes,
                    )
            else:
                await self.put_dir(source.as_posix(), target.as_posix())
        else:
            await self.put_file(source.as_posix(), target.as_posix())

    async def put_dir(self, src, dst):
        """Uploads the contents of the source directory to the target path."""
        src = Path(src)
        dst = Path(dst)

        self.info(f"Starting directory upload: {src} -> {dst}")

        # Skip certain directories
        if src.stem in ["__pycache__", ".idea"]:
            self.info(f"Skipping directory: {src}")
            return

        # Ensure the destination directory exists
        self.info(f"Creating destination directory: {dst}")
        await self.run(f"mkdir -p {dst.as_posix()}")

        # Gather all files and their info
        files_to_check = []
        remote_stats = []
        for item in os.listdir(src):
            if ".git" in item:
                self.info(f"Skipping .git item: {item}")
                continue

            src_path = src / item
            dst_path = dst / item

            if src_path.is_file():
                files_to_check.append((src_path, dst_path))
                self.info(f"Adding file to check: {src_path}")

        # Batch check files
        if files_to_check:
            self.info("Performing batch file check")
            check_cmd = " && ".join(
                [
                    f'stat -c "%s %Y" "{p[1].as_posix()}" 2>/dev/null || echo "NOT_FOUND"'
                    for p in files_to_check
                ]
            )
            self.info(f"Batch check command: {check_cmd}")
            result = await self.run(check_cmd)
            remote_stats = result.stdout.strip().split("\n")
            self.info(f"Received {len(remote_stats)} remote file stats")

        # Prepare files for upload
        files_to_upload = []
        large_files = []
        for (src_path, dst_path), remote_stat in zip(files_to_check, remote_stats):
            file_size = src_path.stat().st_size
            local_mtime = int(src_path.stat().st_mtime)

            self.info(f"Checking file: {src_path}")
            self.info(f"  Local size: {file_size}, Local mtime: {local_mtime}")

            if remote_stat != "NOT_FOUND":
                remote_size, remote_mtime = map(int, remote_stat.split())
                self.info(f"  Remote size: {remote_size}, Remote mtime: {remote_mtime}")
                if file_size == remote_size and local_mtime <= remote_mtime:
                    self.info(f"[gray]{src_path} -> {dst_path} (unchanged)[/gray]")
                    continue
            else:
                self.info("  File not found on remote")

            if file_size > self.max_size:
                self.info(f"  Large file, will handle separately: {src_path}")
                large_files.append((src_path, dst_path))
            else:
                self.info(f"  Adding to upload list: {src_path}")
                files_to_upload.append((src_path, dst_path))

        # Handle large files
        self.info(f"Processing {len(large_files)} large files")
        for src_path, dst_path in large_files:
            if self.config["enable_urls"] and src_path.name in self.urls:
                url = self.urls[src_path.name]
                if url:
                    self.info(f"Downloading large file from URL: {url} -> {dst_path}")
                    await self.run(f"wget {url} -O {dst_path.as_posix()}", log_all=True)
                    if self.config["enable_print_download"]:
                        self.print_download(src_path.name, url, dst_path, url)
                else:
                    self.info(f"[red]<!> Invalid URL '{url}' for {src_path.name}[/red]")
            else:
                self.info(f"[red]<!> File too big, skipping: {src_path.name}[/red]")

        # Batch upload small files
        if files_to_upload:
            self.info(f"Preparing to upload {len(files_to_upload)} files in batch")
            with io.BytesIO() as tar_buffer:
                with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
                    for src_path, _ in files_to_upload:
                        # self.info(f"Adding to tar: {src_path}")
                        tar.add(src_path, arcname=src_path.name)
                tar_buffer.seek(0)

                self.info(f"Uploading tar archive to {dst}")
                await self.run_safe(f"mkdir -p {dst.as_posix()}")
                path = dst / "temp.tar.gz"
                await asyncssh.scp(tar_buffer, (self.connection, f"{path.as_posix()}"))

                self.info("Extracting tar archive on remote")
                await self.connection.run(
                    f"cd {dst.as_posix()} && tar xzf temp.tar.gz && rm temp.tar.gz"
                )

            for src_path, dst_path in files_to_upload:
                self.info(f"[green]{src_path} -> {dst_path}[/green]")

        # Recursively handle subdirectories
        for item in os.listdir(src):
            src_path = src / item
            if src_path.is_dir():
                self.info(f"Recursing into subdirectory: {src_path}")
                await self.put_dir(src_path, dst / item)

        self.info(f"Completed directory upload: {src} -> {dst} [put_dir / scp-tar]")

    async def put_file(self, src, dst):
        """Uploads a file to the target path."""
        assert self.connection

        src = Path(src)
        dst = Path(dst)

        file_size = src.stat().st_size

        # Check if file exists and compare sizes
        result = await self.run(f'stat -c "%s %Y" {dst.as_posix()}', log_cmd=False)
        if result.exit_status == 0:
            remote_size, remote_mtime = map(int, result.stdout.split())
            local_mtime = int(src.stat().st_mtime)

            if file_size == remote_size and local_mtime <= remote_mtime:
                self.info(f"[gray]{src} -> {dst} (unchanged)[/gray]")
                return

        if file_size > self.max_size:
            if self.config["enable_urls"] and src.name in self.urls:
                url = self.urls[src.name]
                if url:
                    await self.run(f"wget {url} -O {dst}", log_all=True)
                    if self.config["enable_print_download"]:
                        self.print_download(src.name, url, dst, url)
                    return
                else:
                    self.info(f"[red]<!> Invalid URL '{url}' for [/red]{src.name}")

            self.info(f"[red]<!> File too big, skipping[/red] {src.name}")
            return

        if self.config["enable_print_upload"]:
            self.print_upload(src.name, src.parent, dst.parent)

        try:
            await self.connection.run(f"mkdir -p {dst.parent.as_posix()}")
            await asyncssh.scp(src, (self.connection, dst.as_posix()))
            self.info(f"[green]{src} -> {dst} [put-file / scp][/green]")
        except asyncssh.Error as e:
            self.error(f"SCP upload failed: {str(e)}")

    async def get_file(self, src, dst):
        """Downloads a file from the target path using SCP."""
        src = Path(src)
        dst = Path(dst)
        try:
            await asyncssh.scp((self.connection, src.as_posix()), dst.as_posix())
            self.info(f"[green]{src} -> {dst}[/green]")
        except asyncssh.Error as e:
            self.error(f"SCP download failed: {str(e)}")

    async def put_rclone(
        self,
        source,
        target,
        forbid_recursive,
        rclone_excludes,
        rclone_includes,
        print_cmd=True,
        print_output=True,
    ):
        start_time = time.time()

        source = Path(source)
        target = Path(target)

        ssh_key_path = os.path.expanduser("~/.ssh/id_rsa")
        # self.info(f"Using SSH key: {ssh_key_path}")

        # Create temporary rclone config file
        async with aiofiles.tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".conf"
        ) as temp_config:
            config_content = f"""[sftp]
	type = sftp
	host = {self.ip}
	port = {self.port}
	user = root
	key_file = {ssh_key_path}
	"""
            await temp_config.write(config_content)
            temp_config_path = temp_config.name
        # self.info(f"Created temporary rclone config at: {temp_config_path}")

        # Prepare rclone flags
        flags = [
            "--checkers",
            "50",  # Increase number of checkers for potentially faster processing
            "--transfers",
            "64",  # Increase number of concurrent transfers
            "--buffer-size",
            "2M",  # Increase buffer size for potentially better performance
            "--stats-one-line",  # Show stats in one line
            "--log-level",
            "INFO",  # Set log level to INFO for detailed output
            "--log-format",
            "{{.Src}}",  # Only show source file names in log output
            # '--no-progress'  # Disable progress display
        ]

        if forbid_recursive:
            flags.extend(["--max-depth", "1"])
        # self.info("Recursive transfer forbidden, max depth set to 1")

        for exclude in rclone_excludes:
            flags.extend(["--exclude", exclude])
        for include in rclone_includes:
            flags.extend(["--include", include])

        # Prepare rclone command
        rclone_cmd = [
            "rclone",
            "--config",
            temp_config_path,
            "copy",
            source.as_posix(),
            f"sftp:{target.as_posix()}",
        ] + flags

        s = f"[green]{source.as_posix()} -> {target.as_posix()} (rclone)[/green]"
        if rclone_excludes:
            self.info(f"exclude: {' '.join(rclone_excludes)}")
        if rclone_includes:
            self.info(f"include: {' '.join(rclone_includes)}")

        if print_cmd:
            # self.info("\n")
            self.info(s)

        # self.info_cmd(" ".join(rclone_cmd), False)

        try:
            if print_output:
                process = await asyncio.create_subprocess_exec(
                    *rclone_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                # Process output in real-time
                async def log_output(stream, prefix):
                    async for line in stream:
                        self.info(f"{prefix}: {line.decode().strip()}")

                await asyncio.gather(
                    log_output(process.stdout, "STDOUT"),
                    log_output(process.stderr, "STDERR"),
                )

            await process.wait()

            if process.returncode != 0:
                self.info(f"Error running rclone. Exit code: {process.returncode}")

        finally:
            os.unlink(temp_config_path)
        # self.info(f"Deleted temporary rclone config: {temp_config_path}")

        end_time = time.time()
        duration = end_time - start_time

    # self.info(f"rclone transfer finished. Duration: {duration:.2f} seconds")

    async def get_rclone(
        self,
        source,
        target,
        forbid_recursive,
        rclone_excludes,
        rclone_includes,
        print_cmd=True,
        print_output=True,
    ):
        source = Path(source)
        target = Path(target)

        ssh_key_path = os.path.expanduser("~/.ssh/id_rsa")

        async with aiofiles.tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".conf"
        ) as temp_config:
            await temp_config.write(f"""[sftp]
	type = sftp
	host = {self.ip}
	port = {self.port}
	user = root
	key_file = {ssh_key_path}
	""")
            temp_config_path = temp_config.name

        flags = ["--progress"]
        if forbid_recursive:
            flags.extend(["--max-depth", "1"])

        for exclude in rclone_excludes:
            flags.extend(["--exclude", exclude])
        for include in rclone_includes:
            flags.extend(["--include", include])

        flags.extend(
            [
                "--update",
                "--use-mmap",
                "--delete-excluded",
                "--checkers",
                "50",
                "--transfers",
                "64",
                "--sftp-set-modtime",
            ]
        )

        rclone_cmd = [
            "rclone",
            "--config",
            temp_config_path,
            "copy",
            "--update",
            f"sftp:{source.as_posix()}",
            target.as_posix(),
            "-v",
        ] + flags

        if print_cmd:
            self.info_cmd(" ".join(rclone_cmd))
            self.info(f"{source} -> {target}")

        try:
            process = await asyncio.create_subprocess_exec(
                *rclone_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                if print_output:
                    self.info(stdout.decode())
            else:
                self.info(f"Error running rclone. Exit code: {process.returncode}")
                self.info("Standard output:")
                self.info(stdout.decode())
                self.info("Standard error:")
                self.info(stderr.decode())
        finally:
            os.unlink(temp_config_path)

    async def put_rsync(
        self,
        source,
        target,
        forbid_recursive=False,
        rsync_excludes=None,
        rsync_includes=None,
        print_cmd=True,
        print_output=True,
    ):
        start_time = time.time()

        source = Path(source)
        target = Path(target)

        # Prepare rsync flags
        flags = [
            "-avz",  # archive mode, verbose, and compress
            "--progress",  # show progress during transfer
            "-e",
            f"ssh -p {self.port} -i ~/.ssh/id_rsa",  # specify SSH options
            "--stats",  # give some file-transfer stats
        ]

        if not forbid_recursive:
            flags.append("--recursive")

        if rsync_excludes:
            for exclude in rsync_excludes:
                flags.extend(["--exclude", exclude])
        if rsync_includes:
            for include in rsync_includes:
                flags.extend(["--include", include])

        # Exclude Python cache files and directories
        flags.extend(
            [
                "--exclude",
                "*.pyc",
                "--exclude",
                "__pycache__",
                "--exclude",
                "*.pyo",
                "--exclude",
                "*.pyd",
            ]
        )

        # Prepare rsync command
        rsync_cmd = (
            [
                "rsync",
            ]
            + flags
            + [
                source.as_posix() + ("/" if source.is_dir() else ""),
                f"root@{self.ip}:{target.as_posix()}",
            ]
        )

        s = f"[green]{source.as_posix()} -> {target.as_posix()} (rsync)[/green]"
        if rsync_excludes:
            self.info(f"exclude: {' '.join(rsync_excludes)}")
        if rsync_includes:
            self.info(f"include: {' '.join(rsync_includes)}")

        if print_cmd:
            self.info(s)
            self.info_cmd(" ".join(rsync_cmd), False)

        try:
            process = await asyncio.create_subprocess_exec(
                *rsync_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            if print_output:

                async def log_output(stream, prefix):
                    async for line in stream:
                        self.info(f"{prefix}: {line.decode().strip()}")

                await asyncio.gather(
                    log_output(process.stdout, "STDOUT"),
                    log_output(process.stderr, "STDERR"),
                )

            await process.wait()

            if process.returncode != 0:
                self.info(f"Error running rsync. Exit code: {process.returncode}")

        except Exception as e:
            self.error(f"Error during rsync operation: {str(e)}")

        end_time = time.time()
        duration = end_time - start_time
        self.info(f"rsync transfer finished. Duration: {duration:.2f} seconds")

    async def get_rsync(
        self,
        source,
        target,
        forbid_recursive=False,
        rsync_excludes=None,
        rsync_includes=None,
        print_cmd=True,
        print_output=True,
    ):
        start_time = time.time()

        source = Path(source)
        target = Path(target)

        # Prepare rsync flags
        flags = [
            "-avz",  # archive mode, verbose, and compress
            "--progress",  # show progress during transfer
            "-e",
            f"ssh -p {self.port} -i ~/.ssh/id_rsa",  # specify SSH options
            "--stats",  # give some file-transfer stats
        ]

        if not forbid_recursive:
            flags.append("--recursive")

        if rsync_excludes:
            for exclude in rsync_excludes:
                flags.extend(["--exclude", exclude])
        if rsync_includes:
            for include in rsync_includes:
                flags.extend(["--include", include])

        flags.extend(["--exclude", ".*/"])

        # Prepare rsync command
        # Add trailing slash for directories (detected by lack of suffix)
        source_path = source.as_posix()
        if "." not in source.name:
            source_path += "/"

        rsync_cmd = (
            [
                "rsync",
            ]
            + flags
            + [
                f"root@{self.ip}:{source_path}",
                target.as_posix(),
            ]
        )

        s = f"[green]{source.as_posix()} -> {target.as_posix()} (rsync)[/green]"
        if rsync_excludes and print_cmd:
            self.info(f"exclude: {' '.join(rsync_excludes)}")
        if rsync_includes and print_cmd:
            self.info(f"include: {' '.join(rsync_includes)}")

        if print_cmd:
            self.info(s)
            self.info_cmd(" ".join(rsync_cmd), False)

        try:
            process = await asyncio.create_subprocess_exec(
                *rsync_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            async def log_output(stream, prefix, is_error):
                if print_output and not is_error:
                    async for line in stream:
                        self.info(f"{prefix}: {line.decode().strip()}")

            await asyncio.gather(
                log_output(process.stdout, "STDOUT", False),
                log_output(process.stderr, "STDERR", True),
            )

            await process.wait()

            if process.returncode != 0:
                self.info(f"Error running rsync. Exit code: {process.returncode}")

        except Exception as e:
            self.error(f"Error during rsync operation: {str(e)}")

        end_time = time.time()
        duration = end_time - start_time
        if print_output:
            self.info(f"rsync transfer finished. Duration: {duration:.2f} seconds")

    def print_upload(self, item, source, target):
        self.info(f"Uploading {os.path.join(source, item)} to {target}")

    def print_download(self, item, source, target, url):
        self.info(f"Downloading {item} from {source} to {target}")

    async def exists(self, path):
        try:
            if isinstance(path, Path):
                path = path.as_posix()
            await self.sftp.lstat(path)
            return True
        except asyncssh.SFTPError:
            return False

    async def mkdir(self, path, mode=511, ignore_existing=False):
        """Augments mkdir by adding an option to not fail if the folder exists"""
        if isinstance(path, Path):
            path = path.as_posix()

        try:
            await self.run(f"mkdir -p {path}")
        except asyncssh.SFTPError:
            if ignore_existing:
                pass
            else:
                raise

    # @staticmethod
    # def print_cmd(cmd):
    #     log.info(f"> {cmd}")

    # @staticmethod
    # def print_header(header):
    #     log.info(f"\n{'=' * 20}\n{header}\n{'=' * 20}")
    def info_cmd(self, text, newline=True):
        if newline:
            self.info("\n")
        self.info(f"> {text}")


# import logging
# import subprocess
# import sys
# from pathlib import Path
#
# import asyncssh
#
# import deploy_utils
# from src.lib.loglib import print_cmd
#
# log = logging.getLogger(__name__)
#
#
# class SSHClient:
#     """
#     Custom asynchronous SSH client for Discore deployments.
#
#     This class provides functionality similar to the original DiscoreSSHClient,
#     but uses asyncssh for asynchronous SSH operations.
#
#     Methods:
#         connect: Establishes an SSH connection.
#         run: Executes a command on the remote instance.
#         file_exists: Checks if a file exists on the remote instance.
#         mount: Mounts a remote directory locally.
#     """
#
#     def __init__(self):
#         self.connection = None
#         self.sftp = None
#
#     async def connect(self, hostname, port, username, password=None, key_filename=None):
#         """Establish an SSH connection."""
#         self.connection = await asyncssh.connect(
#             hostname, port=port,
#             username=username, password=password,
#             client_keys=key_filename,
#             known_hosts=None  # Note: Consider using known_hosts in production
#         )
#         self.sftp = await self.connection.start_sftp_client()
#
#     async def run(self, cm, cwd=None, *, log_all=False):
#         """
#         Execute a command on the remote instance.
#         """
#         cm = cm.replace("'", '"')
#         if cwd is not None:
#             cm = f"cd {cwd}; {cm}"
#
#         cm = f"/bin/bash -c '{cm}'"
#
#         print_cmd(cm, log)
#         result = await self.connection.run(cm, check=True)
#
#         if log_output:
#             log.info(result.stdout)
#
#         return result.stdout
#
#     async def file_exists(self, path):
#         """
#         Check if a file exists on the remote instance.
#         """
#         try:
#             await self.sftp.stat(path)
#             return True
#         except asyncssh.SFTPError:
#             return False
#
#     async def mount(self, local_path, remote_path, ip, port):
#         """
#         Mount a remote directory locally
#         """
#         local_path = Path(local_path)
#         local_path.mkdir(parents=True, exist_ok=True)
#
#         if sys.platform.startswith('win'):
#             log.info("Windows not supported yet.")
#             return
#         elif sys.platform.startswith('darwin'):
#             not_mounted = subprocess.run(f'mount | grep -q "{local_path}"', shell=True).returncode != 0
#             if not_mounted:
#                 mount_cmd = f'sshfs root@{ip}:{remote_path} {local_path} -p {port} -o volname=Discore'
#         else:
#             not_mounted = subprocess.run(f'mountpoint -q {local_path}', shell=True).returncode != 0
#             if not_mounted:
#                 mount_cmd = f'sshfs root@{ip}:{remote_path} {local_path} -p {port}'
#
#         if not_mounted:
#             deploy_utils.print_header("Mounting with sshfs...")
#             result = subprocess.run(mount_cmd, shell=True, capture_output=True, text=True)
#             if result.returncode == 0:
#                 log.info("Mounted successfully!")
#             else:
#                 log.info(f"Failed to mount. Error: {result.stderr}")
#         else:
#             log.info("Already mounted.")
#
#     @staticmethod
#     def print_cmd(cmd):
#         log.info(f"> {cmd}")
#
#     @staticmethod
#     def print_header(header):
#         log.info(f"\n{'=' * 20}\n{header}\n{'=' * 20}")
class SSHState(Enum):
    """
    Refers to the state of the SSH connection.
    """

    NONE = 0
    CONNECTING = 1
    CONNECTED = 2
    CONNECTION_REFUSED = 3
    CONNECTION_ERROR = 4
    CONNECTION_LOST = 5
    PERMISSION_DENIED = 6
    HOST_KEY_NOT_VERIFIABLE = 7
    OS_ERROR = 8
