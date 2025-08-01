import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import asyncssh
from desktop_notifier import DesktopNotifier

from errloom.session import Session
from errloom import paths
from errloom.interop import vast_manager
import errloom.deploy.deploy_constants as const
from errloom.deploy.deploy_constants import ERRLOOM_MAIN_PY
from errloom.deploy.deploy_utils import (
    forget,
    invalidate,
    make_header_text,
    open_terminal,
)
from errloom.deploy.deployment_step import DeploymentStep
from errloom.deploy.remote_view import RemoteView
from errloom.ssh import SSH, SSHState
from errloom.interop.vast_instance import VastInstance
import userconf


class ErrloomRemote_SyncMan:
    """Placeholder class for session synchronization management."""

    def __init__(self, remote):
        self.remote = remote

    async def start(self):
        """Start session synchronization."""
        pass

    async def stop(self):
        """Stop session synchronization."""
        pass

    async def kill(self):
        """Kill synchronization process."""
        pass

    async def is_running(self):
        """Check if synchronization is running."""
        return False


class ErrloomRemote_VLLMMan:
    """Placeholder class for VLLM server management."""

    def __init__(self, remote):
        self.remote = remote

    async def start(self, force_restart=False):
        """Start VLLM server."""
        pass

    async def stop(self):
        """Stop VLLM server."""
        pass

    async def kill(self):
        """Kill VLLM server process."""
        pass

    async def is_running(self):
        """Check if VLLM server is running."""
        return False

    @staticmethod
    def get_server_url(instance):
        """Get the VLLM server URL."""
        if hasattr(instance, "public_ipaddr") and instance.ports.get("8000/tcp"):
            return f"http://{instance.public_ipaddr}:8000"
        return None


@dataclass
class DeploymentStatus:
    """
    A data class to return the overall status of a deployed instance.
    """

    """The state of the SSH connection."""
    connection: SSHState = SSHState.NONE

    """Where are we at with this deployment installation?"""
    installation_step = DeploymentStep.ZERO

    """Are the apt packages installed?"""
    apt_packages_installed = False

    """Is the errloom.py script running on the remote machine? (the python process will be tagged as errloom)"""
    is_errloom_running = False

    """Is the vllm server running on the remote machine? (the python process will be tagged as vllm)"""
    is_vllm_running = False

    """Is the remote mounted?"""
    is_mounted = False

    """What session is this remote currently rendering?"""
    work_session: Optional[str] = ""


class ErrloomRemote:
    install_checkpath = "/root/.errloom_installed"
    apt_checkpath = "/root/.packages_installed"
    WORK_SESSION_FILE = "/workspace/errloom_deploy/session.txt"

    def __init__(self, instance: VastInstance, session: Optional[Session] = None):
        self.instance: VastInstance = instance
        self.session: Optional[Session] = session
        self.ssh: SSH = SSH(self)
        self.connection_state = SSHState.NONE

        self.syncman = ErrloomRemote_SyncMan(self)
        self.vllmman = ErrloomRemote_VLLMMan(self)
        self.notifier = DesktopNotifier()
        self.last_balance = None
        self.watcher = None

        self._log = logging.getLogger("ErrloomRemote")

    @classmethod
    async def ConnectNew(
        cls, instance: VastInstance, session: Optional[Session] = None
    ):
        remote = cls(instance, session)
        await remote.connect()
        return remote

    @property
    def logger(self)->logging.Logger:
        self._log.name = "Errloom"
        if self.instance:
            self._log.name = f"{self.instance.id}"

        return self._log

    async def to_view(self):
        return await RemoteView.from_remote(self)

    def log_info(self, message):
        self.logger.info(message)
        if self.ssh:
            self.ssh.logs.info(message)
        invalidate()

    def log_warning(self, message):
        self.logger.warning(message)
        if self.ssh:
            self.ssh.logs.warning(message)
        invalidate()

    def log_error(self, message):
        self.logger.error(message)
        if self.ssh:
            self.ssh.logs.error(message)
        invalidate()

    async def connect(self):
        assert self.ssh

        self.connection_state = SSHState.CONNECTING
        self.ssh.reset()

        async def fn():
            assert self.ssh

            try:
                # self.info(make_header_text("Connecting through SSH ..."))
                await self.ssh.connect(self.ip, int(self.port), username="root")
                self.log_info("Successfully connected through SSH.")
                self.connection_state = SSHState.CONNECTED
            except ConnectionRefusedError as e:
                self.log_error(f"Connection refused: {e}")
                self.connection_state = SSHState.CONNECTION_REFUSED
                self.refresh_instance_status()
            except asyncssh.ConnectionLost as e:
                self.log_error(f"Failed to connect ({e}).")
                self.connection_state = SSHState.CONNECTION_LOST
            except asyncssh.PermissionDenied as e_denied:
                self.log_warning("Permission denied.")
                self._log.exception(e_denied)
                self.connection_state = SSHState.PERMISSION_DENIED
            except asyncssh.HostKeyNotVerifiable:
                self.log_warning("Host key not verifiable.")
                self.connection_state = SSHState.HOST_KEY_NOT_VERIFIABLE
            except OSError as e:
                self.log_error(f"OSError: {e}")
                self.connection_state = SSHState.OS_ERROR

        await fn()
        if (
            self.connection_state == SSHState.HOST_KEY_NOT_VERIFIABLE
            or self.connection_state == SSHState.PERMISSION_DENIED
        ):
            # Open a terminal to ask for permissions
            # Host key not verifiable: should accept the key (type yes)
            # Permission denied: most likely the ssh agent is locked and the user must enter their password
            await self.shell()
            await fn()
            return

    async def disconnect(self):
        if self.ssh:
            await self.ssh.disconnect()
            self.connection_state = SSHState.NONE

    @property
    def wants_vllm(self):
        return self.session and self.session.res("workflow.json") is not None

    @property
    def src(self) -> Path:
        return paths.root

    @property
    def dst(self) -> Path:
        return Path("/workspace/errloom_deploy")

    @property
    def dst_session(self) -> Path:
        return Path(
            f"/workspace/errloom_deploy/{paths.user_sessions}/{self.session.dirpath.stem}/"
        )

    @property
    def dst_errloom(self) -> Path:
        return self.dst / "errloom"

    @property
    def dst_hf_token(self) -> Path:
        return Path("~/.huggingface/token")

    @property
    def dst_wandb_token(self) -> Path:
        return Path("~/.netrc")

    @property
    def id(self):
        return self.instance.id

    @property
    def ip(self):
        return self.instance.ssh_host

    @property
    def port(self):
        return self.instance.ssh_port

    @property
    def ssh_command(self):
        import shutil

        ssh_path = shutil.which("ssh")
        target_dir = self.dst / "sessions"

        # Break down the tmux commands
        tmux_commands = [
            f'tmux send-keys -t0 "cd {target_dir}" Enter',
            # f'tmux send-keys -t0 "cd $(ls -d */ | sort | head -n1)" Enter',
            f'tmux send-keys -t0 "ls" Enter',
        ]

        # Join commands with && for sequential execution
        tmux_sequence = " && ".join(tmux_commands)

        # Construct final SSH command
        return f"{ssh_path} -t -p {self.port} root@{self.ip} '{tmux_sequence} && bash'"

    def refresh_instance_status(self):
        from errloom.deploy import vast_manager

        info = vast_manager.instance.fetch_instance(self.instance.id)  # TODO
        self.instance = info

        if (
            self.connection_state == SSHState.CONNECTED
            and self.instance.status != "running"
        ):
            raise Exception(
                "Remote has crashed while running for some reason - we need to handle this."
            )

    async def is_errloom_running(self):
        return (
            self.ssh
            and self.ssh.connection
            and await self.ssh.is_process_running(ERRLOOM_MAIN_PY)
        )

    async def is_vllm_running(self):
        return (
            self.ssh
            and self.ssh.connection
            and await self.ssh.is_process_running("vf-vllm")
        )

    async def probe_worksession(self) -> str:
        if not self.ssh:
            return ""

        if not await self.ssh.file_exists(self.WORK_SESSION_FILE):
            return ""

        work_session = await self.ssh.read_file(self.WORK_SESSION_FILE)
        return work_session or ""

    async def has_session(self, session_name: str) -> bool:
        """
        Check if a session exists on the remote instance.

        Args:
            session_name (str): Name of the session to check for

        Returns:
            bool: True if the session exists, False otherwise
        """
        if not self.ssh or not self.ssh.connection:
            return False

        session_path = self.dst / "sessions" / session_name
        return await self.ssh.file_exists(session_path)

    async def set_session(self, session_name: str):
        """
        Set the current work session for the remote.

        This method updates the .work_session file on the remote remote
        with the provided session name. This allows for tracking the
        current active work session across deployments.

        Args:
            session_name (str): The name of the work session to set.

        Raises:
            Exception: If the SSH connection is not established.
        """
        if not self.ssh or not self.ssh.connection:
            raise Exception("SSH connection not established. Cannot set work session.")
        # Create or overwrite the .work_session file with the new session name
        command = f"echo '{session_name}' > {self.WORK_SESSION_FILE}"
        result = await self.ssh.run(command)

        if result.exit_status == 0:
            self.logger.info(f"Set work session to: {session_name}")
        else:
            self.logger.error(f"Failed to set work session. Error: {result.stderr}")

    async def probe_deployment_status(self) -> DeploymentStatus:
        status = DeploymentStatus(connection=self.connection_state)

        if status.connection == SSHState.CONNECTED:
            # Use gather to run async checks concurrently
            is_errloom_running, is_vllm_running, work_session = (
                await self.ssh.is_process_running(ERRLOOM_MAIN_PY),
                await self.ssh.is_process_running("vf-vllm"),
                await self.probe_worksession(),
            )

            status.is_errloom_running = is_errloom_running
            status.is_vllm_running = is_vllm_running
            status.work_session = work_session
            status.installation_step = await self.probe_installation_step()
            status.apt_packages_installed = await self.probe_apt_packages_installed()

        # Uncomment if needed:
        # status.is_mounted = await self.ssh.is_mounted()

        return status

    async def is_ready(self) -> bool:
        from errloom.deploy import vast_manager

        try:
            instances = await vast_manager.instance.fetch_instances()
            info = next((i for i in instances if i.id == self.id), None)

            if info:
                self.instance = info
                return self.instance.status == "running"

            return False

        except Exception as e:
            self.log_error(f"Error while checking instance readiness: {str(e)}")
            return False

    async def deploy(self, redeploy=False):
        assert self.instance

        # if not self.instance.status == "running":
        #     self.log_error(f"Cannot deploy: instance is not running. (status={self.instance.status})")
        #     return

        if not self.ssh.is_connected():
            self.log_error("Not connected")
            return

        await vast_manager.instance.wait_for_ready(self.instance)

        src = self.src
        dst = self.dst
        step = await self.probe_installation_step()  # zero, apt, git, optional, done

        if self.session:
            await self.set_session(self.session.name)

        # Set NUMBA_CACHE_DIR to a writable directory
        # ----------------------------------------
        if redeploy:
            step = DeploymentStep.ZERO
            self.log_info(f"[red]Deploment: remove old deployment in 3 seconds to start over ({dst.as_posix()})...[/red]")
            await asyncio.sleep(3)
            await self.ssh.run(f"rm -rf {dst.as_posix()}")

        self.logger.info(
            make_header_text(
                f"""Deploying errloom to {self.ip} ...
Detected installation step: {step.name} ({step.value} / {DeploymentStep.DONE.value})"""
            )
        )

        # Send API tokens
        # ----------------------------------------
        # Get tokens from userconf with fallbacks
        hf_token = getattr(userconf, 'hf_token', '')
        wandb_token = getattr(userconf, 'wandb_token', '')

        # Setup Hugging Face token
        await self.ssh.run(f"mkdir -p ~/.huggingface")
        await self.ssh.run(f"echo '{hf_token}' > {self.dst_hf_token.as_posix()}")

        # Setup W&B token
        netrc_entry = f"machine api.wandb.ai\n  login user\n  password {wandb_token}\n"
        await self.ssh.run(f"echo '{netrc_entry}' > {self.dst_wandb_token.as_posix()}")
        await self.ssh.run(f"chmod 600 {self.dst_wandb_token.as_posix()}")

        if not self.probe_apt_packages_installed():
            self.logger.info(make_header_text("[1/3] Installing system packages ..."))
            await self.apt_install()

        # Set NUMBA_CACHE_DIR to a writable directory
        # ----------------------------------------
        if not await self.ssh.file_exists(dst / "numba_cache"):
            numba_cache_dir = dst / "numba_cache"
            await self.ssh.run(f"mkdir -p {numba_cache_dir}")
            await self.ssh.run(f"export NUMBA_CACHE_DIR={numba_cache_dir}")

            # Add the export to .bashrc to persist across sessions
            bashrc_path = "~/.bashrc"
            await self.ssh.run(
                f"echo 'export NUMBA_CACHE_DIR={numba_cache_dir}' >> {bashrc_path}"
            )

            self.logger.info(f"Set NUMBA_CACHE_DIR to {numba_cache_dir}")

        if step.value <= DeploymentStep.GIT.value:
            self.log_info(make_header_text("[2/3] Cloning repositories ..."))
            await self.run_git_clones(dst)

            self.log_info(make_header_text("[2.5/3] Sending fast uploads ..."))
            await self.send_fast_uploads(dst, src)

        # Install requirements.txt for errloom and thauten
        if step.value < DeploymentStep.DONE.value:
            self.logger.info(make_header_text("[3/3] Installing dependencies ..."))
            await self.pip_upgrade()

        # Mark the installation
        if step.value < DeploymentStep.DONE.value:
            self.logger.info(make_header_text("[DONE] Marking installation as done ..."))
            self.logger.info("\n")
            await self.ssh.run_safe(f"touch {ErrloomRemote.install_checkpath}")

        self.log_info("Deployment done")

    async def apt_install(self):
        for apt_package in const.APT_PACKAGES:
            # self.log.info(f"Installing {apt_package}...")
            try:
                await self.ssh.run(f"apt-get install {apt_package} -y")
            except Exception as e:
                self.log_error(f"Failed to install {apt_package}, skipping ...")
                self.log_error(e)

        # Mark the installation
        await self.ssh.run(f"touch {ErrloomRemote.apt_checkpath}")

    async def probe_apt_packages_installed(self) -> bool:
        if await self.ssh.file_exists(self.apt_checkpath):
            return True
        return False

    async def probe_installation_step(self) -> DeploymentStep:
        if await self.ssh.file_exists(ErrloomRemote.install_checkpath):
            return DeploymentStep.DONE
        if await self.ssh.file_exists(self.dst):
            return DeploymentStep.GIT

        return DeploymentStep.ZERO

    async def pip_upgrade(self):
        src = self.src
        dst = self.dst

        # Fix a potential annoying warning ("There was an error checking the latest version of pip")
        await self.ssh.run("rm -r ~/.cache/pip/selfcheck/")
        await self.ssh.put_file(
            self.src / "requirements-vastai.txt", dst / "requirements.txt"
        )

        # Update pip
        await self.ssh.run(
            f"{const.VAST_PYTHON_BIN} -m pip install --upgrade pip --force-reinstall",
            log_all=False,
        )

        # Install flash-attn first (as shown in notebook)
        await self.ssh.run(
            f"{const.VAST_PIP_BIN} install flash-attn==2.8.0.post2 --no-build-isolation --break-system-packages",
            log_all=True,
        )

        # Install vllm
        await self.ssh.run(
            f"{const.VAST_PIP_BIN} install vllm==0.9.1 --extra-index-url https://download.pytorch.org/whl/cu128",
            log_all=True,
        )

        # Install other dependencies
        await self.ssh.run(
            f"{const.VAST_PIP_BIN} install wandb huggingface_hub",
            log_all=True,
        )

        # Install errloom requirements
        await self.ssh.run(f"cd {dst}; pip install -r requirements.txt", log_all=True)

        # Install errloom in development mode
        if await self.ssh.file_exists(self.dst_errloom / "setup.py"):
            await self.ssh.run(
                f"cd {self.dst_errloom}; {const.VAST_PIP_BIN} install -e . --break-system-packages",
                log_all=True,
            )

    def get_mount_path(self) -> Path:
        return Path(userconf.vastai_sshfs_path).expanduser() / "errloom_deploy"

    async def is_mounted(self) -> bool:
        return await self.ssh.is_mounted(self.get_mount_path())

    async def mount(self) -> Path:
        mount_path = self.get_mount_path()
        await self.ssh.mount(
            mount_path, "/workspace/errloom_deploy", self.ip, self.port
        )
        return mount_path

    async def unmount(self):
        await self.ssh.unmount(self.get_mount_path())

    async def shell(self):
        await open_terminal(self.ssh_command)
        invalidate()

    async def send_session(self, session: Optional[Session] = None):
        if session is None:
            session = self.session

        self.session = session

        if not session:
            self.log_error("No session to send")
            return

        dst_path = self.dst / "sessions" / session.dirpath.stem

        if session is not None and session.dirpath.exists():
            self.logger.info(
                make_header_text(f"Syncing session '{session.dirpath.stem}' to remote")
            )
            await self.ssh.mkdir(str(dst_path))
            await self.ssh.put_any(
                session.dirpath,
                dst_path,
                forbid_recursive=True,
                rsync_excludes=[
                    *["*" + e for e in paths.image_exts],
                    ".demucs",
                    ".scripts",
                    "scripts",
                    ".init",
                ],
            )

            # Send the last 50 frames (in case we need them for effects)
            # last = src_session.f_last
            # if last > 1:
            #     for i in range(max(last - 50, 0), last):
            #         src = src_session.det_frame_path(i)
            #         dst = dst_path / 'frames' / f"{src.stem}.jpg"
            #         print(src, dst)
            #         await self.ssh.put_any(src, dst)

            # Send the latest image (no need to know everything before)
            # We could send the last few images as well, that could be useful
            if session.f_last > 0 and session.f_last_path:
                self.log_info(f"Sending last image: {session.f_last_path}")
                dst_file =  dst_path / 'frames' / session.f_last_path.name
                await self.ssh.put_any(session.f_last_path, dst_file)

        invalidate()

    async def send_slow_uploads(self, dst=None, src=None):
        self.log_info(make_header_text("d) Sending slow uploads ..."))

        src = src or self.src
        dst = dst or self.dst
        await self.send_files(src, dst, const.SLOW_UPLOADS, is_ftp=False)
        invalidate()

    async def send_fast_uploads(self, dst=None, src=None):
        src = src or self.src
        dst = dst or self.dst
        await self.send_files(
            src,
            dst,
            const.FAST_UPLOADS,
            rclone_includes=[f"*{v}" for v in paths.text_exts],
            is_ftp=False,
        )

        invalidate()

    async def send_files(
        self,
        src: Path,
        dst: Path,
        file_list: List[str | Tuple[str, str]],
        is_ftp: bool = False,
        rclone_includes: Optional[List[str]] = None,
    ):
        def process_file_paths(
            file_item: Union[str, Tuple[str, str]],
        ) -> Tuple[Path, Path]:
            if isinstance(file_item, tuple):
                source_path, dest_path = file_item
            else:
                source_path = dest_path = file_item

            return normalize_path(src, source_path), normalize_path(dst, dest_path)

        def normalize_path(base_path: Path, file_path: str) -> Path:
            path = Path(file_path)
            if path.is_absolute():
                return path.relative_to(paths.root)
            return base_path / path

        async def transfer_file(source: Path, destination: Path):
            kwargs = (
                {"forbid_rclone": True}
                if is_ftp
                else {"rclone_includes": rclone_includes or []}
            )
            await self.ssh.put_any(source, destination, **kwargs)

        for it in file_list:
            a, b = process_file_paths(it)
            await transfer_file(a, b)

        invalidate()

    async def run_git_clones(self, dst: Path):
        # Clone the repositories
        # ----------------------------------------
        repos = self.get_git_clones()

        self.logger.info(make_header_text("Cloning repositories ..."))
        self.logger.info(str(repos))

        tasks = []
        for target_dir, repo_list in repos.items():
            for repo in repo_list:
                if "errloom" in target_dir.as_posix():
                    await self.ssh.clone_repo(repo, target_dir / Path(repo).stem)
                else:
                    await self.ssh.clone_repo(repo, target_dir)
            # task = asyncio.create_task()
            # tasks.append(task)

        # await asyncio.gather(*tasks)
        await self.ssh.run(
            f"git -C {dst.as_posix()} submodule update --init --recursive", log_all=True
        )

        # Make errloom.py executable
        errloom_py = dst / ERRLOOM_MAIN_PY
        await self.ssh.run_safe(f"chmod +x {errloom_py.as_posix()}")

        invalidate()

    def get_git_clones(self):
        repos = {
            self.dst: ["https://github.com/holo-q/errloom"],
        }

        return repos

    async def _errloom_job(self, upgrade=False, install=False, trace=False):
        cmd = self.get_errloom_command(self.session.name)
        if upgrade or install:
            cmd += " --install"
        if trace:
            cmd += " --trace"

        self.logger.info(make_header_text("Launching errloom for work ..."))
        await self.ssh.run(cmd, log_all=True)
        self.logger.info(make_header_text("Errloom job has ended."))

    def get_errloom_command(self, session="", bin=ERRLOOM_MAIN_PY, run=False):
        dst = Path("/workspace/errloom_deploy")
        dst_main_py = dst / bin
        cmd = f"cd {dst.as_posix()}; {const.VAST_PYTHON_BIN} {dst_main_py.as_posix()}"

        if self.session:
            cmd += f" {session}"

        cmd += " -cli --remote --no_venv -start"
        if run:
            cmd += " --run"
        # cmd += " --trace"
        return cmd

    async def run_errloom(self):
        self.logger.info("(remote:run_errloom) Starting errloom ...")

        forget(self.syncman.start())

        self.logger.info("(remote:run_errloom) killing previous errloom process ...")
        await self.ssh.kill_process(ERRLOOM_MAIN_PY)

        # Verify session exists on remote, send if needed
        if not await self.has_session(self.session.name):
            self.log_info(
                make_header_text("(remote:run_errloom) Session not found on remote, sending first...")
            )
            await self.send_session()

        # Start vllm if it is used by this session
        if self.session and self.session.res('workflow.json') is not None: # TODO centralized vllm detection for sessions
            await self.vllmman.start()

        self.logger.info("(remote:errloom) starting errloom job ...")
        await self._errloom_job()


    async def stop_errloom(self):
        await self.syncman.stop()
        await self.ssh.kill_process(ERRLOOM_MAIN_PY)

    async def start_vllm(self):
        await self.vllmman.start()

    async def stop_vllm(self):
        await self.vllmman.stop()
