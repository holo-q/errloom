import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Union, Dict, List

from prompt_toolkit import Application, HTML
from prompt_toolkit.filters import to_filter, utils, Filter, FilterOrBool
from prompt_toolkit.layout import Window, ConditionalContainer, Float, Container
from prompt_toolkit.widgets import TextArea

from src import paths

logger = logging.getLogger(__name__)


# TODO pull request
def to_filter(bool_or_filter: FilterOrBool) -> Filter:
    """
    Accept both booleans and Filters as input and
    turn it into a Filter.
    """
    if isinstance(bool_or_filter, bool):
        return utils._bool_to_filter[bool_or_filter]

    if isinstance(bool_or_filter, Filter):
        return bool_or_filter

    raise TypeError(f"Expecting a bool or a Filter instance. Got {bool_or_filter!r}")


utils.to_filter = to_filter

global_task_list = []


def get_git_remote_urls(directory: Union[str, Path]) -> List[Dict[str, str]]:
    directory = Path(directory).resolve()
    results = []

    for subdir in directory.iterdir():
        if subdir.is_dir():
            git_dir = subdir / '.git'
            if git_dir.exists():
                try:
                    url = subprocess.check_output(
                        ['git', 'config', '--get', 'remote.origin.url'],
                        cwd=subdir,
                        universal_newlines=True
                    ).strip()

                    results.append({
                        'Folder': subdir.name,
                        'URL': url
                    })
                except subprocess.CalledProcessError as e:
                    # Git command failed, skip this directory
                    logger.warning(f"Git command failed, skipping {subdir.name} ({e})")
                    pass

    return results


def forget(coroutine):
    async def wrapped_coroutine():
        try:
            await coroutine
        except Exception as e:
            logging.error(f"Background task raised an exception: {e}")
            logging.exception(e)
            print(e)

    task = asyncio.create_task(wrapped_coroutine())
    global_task_list.append(task)
    return task


def is_control_visible(app: Application, control_to_find) -> bool:
    def traverse(container):
        if isinstance(container, Window):
            if container.content == control_to_find:
                return True
        elif isinstance(container, ConditionalContainer):
            if to_filter(container.filter)():
                return traverse(container.content)
        elif isinstance(container, Float):
            if traverse(container.content):
                return True
        elif isinstance(container, Container):
            for c in container.get_children():
                if traverse(c):
                    return True
        return False

    return traverse(app.layout.container)


def bold_character(word: str, index: int) -> HTML:
    """
    Returns an HTML object with the character at the specified index in bold.

    :param word: The word to modify
    :param index: The index of the character to make bold
    :return: HTML object with formatted text
    """
    if index < 0 or index >= len(word):
        return HTML(word)  # Return the word as is if index is out of range

    before = word[:index]
    char = word[index]
    after = word[index + 1:]

    return HTML(f"{before}<b>{char}</b>{after}")


def run_vast_command(command):
    vastpath = paths.root / 'vast'
    return subprocess.check_output([sys.executable, vastpath.as_posix()] + command).decode('utf-8')


def get_os():
    if sys.platform.startswith('win'):
        return 'windows'
    elif sys.platform.startswith('darwin'):
        return 'macos'
    else:
        return 'linux'


async def find_windows_terminal():
    terminals = [
        ("Command Prompt", "cmd.exe", "cmd"),
        ("WezTerm", "wezterm-gui.exe", "wezterm", "start"),
        ("Alacritty", "alacritty.exe", "alacritty"),
        ("Windows Terminal", "wt.exe", "wt"),
        ("ConEmu", "ConEmu64.exe", "conemu"),
        ("Cmder", "Cmder.exe", "cmder"),
        ("PowerShell", "powershell.exe", "powershell"),
    ]

    for name, executable, cmd, *args in terminals:
        path = shutil.which(executable)
        if path:
            return name, cmd, args

    return None, "cmd", []

async def open_terminal(ssh_cmd):
    if sys.platform.startswith('win'):
        terminal_name, terminal_cmd, terminal_args = await find_windows_terminal()

        if terminal_name:
            print(f"Using {terminal_name}")

            if terminal_name == "WezTerm":
                full_cmd = [terminal_cmd] + terminal_args + ["ssh", ssh_cmd]
            elif terminal_name == "Alacritty":
                full_cmd = [terminal_cmd, "-e", "ssh", ssh_cmd]
            elif terminal_name == "Windows Terminal":
                full_cmd = [terminal_cmd, "ssh", ssh_cmd]
            elif terminal_name in ["ConEmu", "Cmder"]:
                full_cmd = [terminal_cmd, "/cmd", ssh_cmd]
            elif terminal_name in ["PowerShell", "Command Prompt"]:
                full_cmd = [terminal_cmd, "/c", "start", terminal_cmd, "/k", ssh_cmd]
            else:
                full_cmd = [terminal_cmd, "/c", ssh_cmd]

            process = await asyncio.create_subprocess_exec(
                *full_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
        else:
            print("No supported terminal found. Falling back to cmd.")
            process = await asyncio.create_subprocess_shell(
                f'start cmd /k {ssh_cmd}',
                shell=True,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
    elif sys.platform.startswith('darwin'):
        # For macOS
        process = await asyncio.create_subprocess_exec(
            'open', '-a', 'Terminal', ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
    else:
        env = os.environ.copy()
        env['TERM'] = 'xterm-256color'
        process = await asyncio.create_subprocess_shell(
            f'kitty -e env TERM=xterm-256color {ssh_cmd}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

    # Wait for the process to complete
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        print(f"Error opening terminal: {stderr.decode().strip()}")

    return process.returncode


def open_shell(ssh):
    import interactive
    print_header("user --shell")
    # Start a ssh shell for the user
    channel = ssh.invoke_shell()
    interactive.interactive_shell(channel)


def print_header(string):
    print("")
    print("----------------------------------------")
    print(f"[green]{string}[/green]")
    print("----------------------------------------")
    print("")


def make_header_text(string):
    s = ""
    s += "\n"
    s += "----------------------------------------\n"
    s += f"[green]{string}[/green]\n"
    s += "----------------------------------------\n"
    # s += "\n"
    return s


def download_vast_script():
    os_type = get_os()
    vastpath = paths.root / 'vast'
    if not vastpath.is_file():
        if os_type == 'windows':
            subprocess.run(["powershell", "-Command", f"Invoke-WebRequest -Uri https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -OutFile {vastpath}"], check=True)
        else:
            os.system(f"wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O {vastpath}; chmod +x {vastpath};")


def invalidate():
    from errloom.deploy import app
    app.invalidate()


def scroll_to_end(text_area: TextArea):
    # Get the underlying Buffer
    buffer = text_area.buffer

    # Set the cursor position to the end of the text
    buffer.cursor_position = len(buffer.text)


def is_scrolled_to_bottom(text_area: TextArea) -> bool:
    # Get the underlying Buffer and Window
    buffer = text_area.buffer
    window = text_area.window

    if not window.render_info:
        return False

    # Get the height of the visible area
    visible_line_count = window.render_info.window_height

    # Get the total number of lines in the buffer
    total_line_count = len(buffer.document.lines)

    # Get the current top visible line
    top_visible_line = window.vertical_scroll

    # Check if we're at the bottom
    return (top_visible_line + visible_line_count) >= total_line_count

def setup_session_logging(session : 'CommandSession', main_loggers = []):
    # Add a handler to the root logger that pipes text to session.add_text
    class SessionLogHandler(logging.Handler):
        def __init__(self, session):
            super().__init__()
            self.session = session

        def emit(self, record):
            log_entry = self.format(record)
            self.session.add_text(log_entry, channel='logging')
            if record.name in main_loggers:
                self.session.add_text(log_entry, channel='main')

    session_handler = SessionLogHandler(session)
    session_handler.setLevel(logging.INFO)
    session_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(session_handler)