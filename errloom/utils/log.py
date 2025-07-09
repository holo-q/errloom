import picologging as logging
import os
from functools import wraps
from threading import local

import rich.traceback
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

cl = Console()
rich.traceback.install(
    show_locals=False,  # You have False - but True is great for debugging
    word_wrap=True,  # ✓ You already have this
    extra_lines=2,  # Show more context lines around the error
    max_frames=10,  # Limit very deep stacks (default is 100)
    indent_guides=True,  # Visual indentation guides
    locals_max_length=10,  # Limit local var representation length
    locals_max_string=80,  # Limit string local var length
    locals_hide_dunder=True,  # Hide __dunder__ variables in locals
    locals_hide_sunder=True,  # Hide _private variables in locals
    suppress=[],  # You can add modules to suppress here
)

# Global indent tracker - thread-local to handle concurrent logging
_indent_state = local()

def _get_indent_level() -> int:
    """Get current indent level, defaulting to 0."""
    return getattr(_indent_state, 'level', 0)

def _set_indent_level(level: int) -> None:
    """Set current indent level."""
    _indent_state.level = level

def indent(func):
    """
    Decorator that increments log indentation for the duration of the function.

    Usage:
        @indent
        def my_function():
            log("This will be indented")
            # ... function body
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        current_level = _get_indent_level()
        _set_indent_level(current_level + 1)
        try:
            return func(*args, **kwargs)
        finally:
            _set_indent_level(current_level)
    return wrapper

def push_indent(i=1):
    _set_indent_level(_get_indent_level() + i)

def pop_indent(i=1):
    _set_indent_level(_get_indent_level() - i)

class CustomRichHandler(RichHandler):
    """
    A custom RichHandler that formats log messages to include the
    class and function name of the caller.
    """

    def __init__(self, *kargs, print_path=False, highlight=True, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.print_path = print_path
        self.highlight = highlight

    def render_message(self, record: logging.LogRecord, message: str):
        """
        Renders the log message with a prefix containing the caller's context.
        """

        path = ""
        if self.print_path:
            classname = getattr(record, "classname", None)
            if classname:
                path = f"({classname}.{record.funcName})"
            else:
                filename = os.path.basename(record.pathname).replace(".py", "")
                path = f"({filename}.{record.funcName})"

        # For multiline messages, place them on a new line after the prefix
        # msg = None
        # if isinstance(record.msg, str):
        #     msg = record.msg
        # elif isinstance(record.msg, Table):
        #     msg = PrintedText(record.msg).markup
        # elif isinstance(record.msg, Text):
        #     msg = record.msg.markup
        # elif isinstance(record.msg, RenderableType):
        #     msg = str(record.msg)
        # elif isinstance(record.msg, Panel):
        #     msg = PrintedText(record.msg).markup
        # else:
        #     # raise ValueError(f"Unsupported message type: {type(record.msg)}")

        msg = record.msg
        msgtext = PrintedText(msg, highlight=self.highlight).markup
        assert msg is not None

        # Compute the whole prefix so we can measure it
        lvl = _get_indent_level()
        prefix_indent = f"{'... ' * lvl}"
        prefix_path = ""

        if self.print_path:
            prefix_path = f"{path}: "

        # Measure the prefix
        idt = lvl * 2 # len(prefix_indent)
        if self.print_path and "\n" in msgtext:
            idt += len(prefix_path)

        # Render to text
        msgtext = PrintedText(msgtext, highlight=False).markup # TODO idt needs to also contain the prefix created by the logging library (timestamp & level)

        # everything after the first line needs to be pushed horizontally to align
        if "\n" in msgtext:
            lines = msgtext.split("\n")
            msgtext = ""
            if lines[0].strip():
                msgtext = lines[0] + "\n"
            else:
                lines = lines[1:]
            msgtext += "\n".join(" " * (idt + 1) + line for line in lines[1:])

        full = f"[dim]{prefix_indent} {prefix_path}[/dim]{msgtext}"
        return Text.from_markup(full)


def setup_logging(
    level: str = "DEBUG",
    print_path: bool = True,
    highlight: bool = True
) -> None:
    """
    Setup basic logging configuration for the errloom package.

    Args:
        :param print_path:
        :param level: The logging level to use. Defaults to "DEBUG".
    """

    # Get the root logger for the package
    logger = logging.getLogger()
    logger.setLevel(level.upper())

    # if logger.hasHandlers():
    #     logger.handlers.clear()

    # Create a RichHandler
    if not any(isinstance(handler, CustomRichHandler) for handler in logger.handlers):
        handler = CustomRichHandler(rich_tracebacks=True, show_path=False, console=cl, markup=True, print_path=print_path, highlight=highlight)
        logger.addHandler(handler)

    # Prevent the logger from propagating messages to the root logger
    # logger.propagate = False


def print_prompt_completions_sample(
    prompts: list[str],
    completions: list[dict],
    rewards: dict[str, list[float]],
    step: int,
    num_samples: int = 1,  # Number of samples to display
) -> None:
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    table.add_column("Reward", style="bold cyan", justify="right")

    # Get the reward values from the dictionary
    reward_values = rewards.get("reward", [])

    # Ensure we have rewards for all prompts/completions
    if len(reward_values) < len(prompts):
        # Pad with zeros if we don't have enough rewards
        reward_values = reward_values + [0.0] * (len(prompts) - len(reward_values))

    # Only show the first num_samples samples
    samples_to_show = min(num_samples, len(prompts))

    for i in range(samples_to_show):
        prompt = prompts[i]
        completion = completions[i]
        reward = reward_values[i]

        # Format prompt (can be string or list of dicts)
        formatted_prompt = Text()
        if isinstance(prompt, str):
            formatted_prompt = Text(prompt)
        elif isinstance(prompt, list):
            # For chat format, only show the last message content (typically the user's question)
            if prompt:
                last_message = prompt[-1]
                content = last_message.get("content", "")
                formatted_prompt = Text(content, style="bright_yellow")
            else:
                formatted_prompt = Text("")
        else:
            formatted_prompt = Text(str(prompt))

        # Create a formatted Text object for completion with alternating colors based on role
        formatted_completion = Text()

        if isinstance(completion, dict):
            # Handle single message dict
            role = completion.get("role", "")
            content = completion.get("content", "")
            style = "bright_cyan" if role == "assistant" else "bright_magenta"
            formatted_completion.append(f"{role}: ", style="bold")
            formatted_completion.append(content, style=style)
        elif isinstance(completion, list):
            # Handle list of message dicts
            for i, message in enumerate(completion):
                if i > 0:
                    formatted_completion.append("\n\n")

                role = message.get("role", "")
                content = message.get("content", "")

                # Set style based on role
                style = "bright_cyan" if role == "assistant" else "bright_magenta"

                formatted_completion.append(f"{role}: ", style="bold")
                formatted_completion.append(content, style=style)
        else:
            # Fallback for string completions
            formatted_completion = Text(str(completion))

        table.add_row(formatted_prompt, formatted_completion, Text(f"{reward:.2f}"))
        if i < samples_to_show - 1:  # Don't add section after last row
            pass  # table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)


def ellipse(text: str, max_length: int = 50) -> str:
    """
    Truncates a string to a maximum length and adds an ellipsis if it's too long.
    """
    if len(text) > max_length:
        return text[:max_length] + '...'
    return text


def PrintedText(renderable, prefix_cols=None, width=None, highlight=True) -> Text:
    """
    Renders any rich renderable to a Text object that can be safely logged.
    This captures the output and converts it to ANSI text, so it can be
    included in standard logging messages without breaking formatting.
    """
    import shutil
    if prefix_cols:
        width, _ = shutil.get_terminal_size()
        width = width - prefix_cols - 1

    if not width:
        width = 120
    c = Console(width=width)
    with c.capture() as capture:
        c.print(renderable, highlight=highlight)
    return Text.from_ansi(capture.get())

logger_main = logging.getLogger("main")

def log(*s, logger=logger_main):
    if not s:
        s = ""
    logger.info(*s)

def logc(logger=logger_main):
    cl.clear()

def logl(*s, logger=logger_main):
    logger.info(*s)
    logger.info("")

def logi(text, logger=logger_main):
    logger.info(f"[cyan]{text}[/]")
    logger.info("")

class LogContext:
    """Context manager for wrapping code blocks with intro/outro logging."""

    def __init__(self, intro_msg: str, outro_msg: str = None, logger=logger_main):
        self.intro = intro_msg
        self.outro = outro_msg or intro_msg.replace("Starting", "Completed").replace(
            "Initializing", "Initialized"
        )
        self.logger = logger

    def __enter__(self):
        log(f"[cyan]{self.intro}[/]", logger=self.logger)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            logl(f"[green]✓[/] {self.outro}", logger=self.logger)
        else:
            logl(f"[red]✗[/] {self.outro} failed: {exc_val}", logger=self.logger)
            return False



