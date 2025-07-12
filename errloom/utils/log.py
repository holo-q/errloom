import logging
import os
from functools import wraps
from math import ceil
from threading import local
from typing import Callable, Optional

import rich.traceback
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

logger = logging.getLogger(__name__)

# ENHANCED LOGGER CLASS
# ----------------------------------------

class EnhancedLogger(logging.Logger):
    """Logger with enhanced push/pop indent functionality."""

    def push(self, a1=1, a2=None, log_func: Optional[Callable] = None):
        """Push indentation with optional logging function."""
        return push(a1, a2, log_func=log_func or self.info)

    def pop(self):
        """Pop indentation from the stack."""
        return pop()

    def indent_ctx(self, a1=1, a2=None, log_func: Optional[Callable] = None):
        """Context manager for temporary indentation."""
        return IndentContext(a1, a2, log_func=log_func or self.info)

    def push_debug(self, a1=1, a2=None):
        """Push with debug level logging."""
        return push(a1, a2, log_func=self.debug)

    def push_info(self, a1=1, a2=None):
        """Push with info level logging."""
        return push(a1, a2, log_func=self.info)

    def push_warning(self, a1=1, a2=None):
        """Push with warning level logging."""
        return push(a1, a2, log_func=self.warning)

    def push_error(self, a1=1, a2=None):
        """Push with error level logging."""
        return push(a1, a2, log_func=self.error)

    def push_critical(self, a1=1, a2=None):
        """Push with critical level logging."""
        return push(a1, a2, log_func=self.critical)

# MAIN SETUP
# ----------------------------------------

cl = Console()
rich.traceback.install(
    show_locals=False,  # You have False - but True is great for debugging
    word_wrap=True,  # ✓ You already have this
    extra_lines=2,  # Show more context lines around the error
    max_frames=10,  # Limit very deep stacks (default is 100)
    indent_guides=True,  # Visual indentation guides
    # locals_max_length=10,  # Limit local var representation length
    # locals_max_string=80,  # Limit string local var length
    # locals_hide_dunder=True,  # Hide __dunder__ variables in locals
    # locals_hide_sunder=True,  # Hide _private variables in locals
    suppress=[],  # You can add modules to suppress here
)


def setup_logging(
    level: str = "DEBUG",
    print_path: bool = True,
    highlight: bool = True
) -> None:
    """
    Setup basic logging configuration for the errloom package.

    Args:
        :param highlight:
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
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)

def getLogger(name: Optional[str] = None) -> EnhancedLogger:
    """
    Get an enhanced logger instance with push/pop indent functionality.

    Args:
        name: Logger name. If None, uses the calling module's name.

    Returns:
        EnhancedLogger instance with push/pop methods.
    """
    if name is None:
        # Get the calling module's name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'unknown')
        else:
            name = 'unknown'

    # Set the logger class before getting the logger
    old_class = logging.getLoggerClass()
    logging.setLoggerClass(EnhancedLogger)

    try:
        logger = logging.getLogger(name)
    finally:
        # Restore the original logger class
        logging.setLoggerClass(old_class)

    return logger



# MAIN
# ----------------------------------------

logger_main = logging.getLogger("main")

def log(*s, logger=logger_main, stacklevel=1):
    if not s:
        s = ""
    logger.info(*s, stacklevel=stacklevel + 1)

def logc(logger=logger_main):
    cl.clear()

def logl(*s, logger=logger_main, stacklevel=1):
    logger.info(*s, stacklevel=stacklevel + 1)
    logger.info("")

def logi(text, logger=logger_main, stacklevel=1):
    logger.info(f"[cyan]{text}[/]", stacklevel=stacklevel + 1)
    logger.info("")

class LogContext:
    """Context manager for wrapping code blocks with intro/outro logging."""

    def __init__(self, intro_msg: str, outro_msg: Optional[str] = None, logger=logger_main):
        self.intro = intro_msg
        self.outro = outro_msg or intro_msg.replace("Starting", "Completed").replace(
            "Initializing", "Initialized"
        )
        self.logger = logger

    def __enter__(self):
        log(f"[cyan]{self.intro}[/]", logger=self.logger, stacklevel=2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            logl(f"[green]✓[/] {self.outro}", logger=self.logger, stacklevel=2)
        else:
            logl(f"[red]✗[/] {self.outro} failed: {exc_val}", logger=self.logger, stacklevel=2)
            return False

# GLOBAL INDENT TRACKER - thread-local to handle concurrent logging
# ----------------------------------------

_indent_state = local()

def _get_indent_stack() -> list[str]:
    """Get current indent stack, defaulting to an empty list."""
    if not hasattr(_indent_state, 'stack'):
        _indent_state.stack = []
    return _indent_state.stack

def indent_decorator(func=None, *, a1=1, a2=None, log_func=None):
    if log_func is None:
        log_func = logger_main.info

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            push(a1, a2, log_func=log_func)
            try:
                return fn(*args, **kwargs)
            finally:
                pop()

        return wrapper

    if callable(func):
        return decorator(func)
    else:
        # Handle @indent("string")
        if func is not None:
            a1 = func
            a2 = ""
        return decorator


def push(a1=1, a2=None, log_func=None):
    """Pushes an indentation string onto the stack."""
    if log_func is None:
        log_func = logger_main.info

    stack = _get_indent_stack()
    seg = ".. "
    segw = len(seg)
    if isinstance(a1, int):
        stack.append(seg * a1)
    elif isinstance(a1, str) and a2 is not None:
        log_func(f"{a1} {a2}")
        # Push spaces to align subsequent logs under the message
        stack.append(a1 + " ")
    elif isinstance(a1, str):
        log_func(a1)
        i = ceil(len(a1) / segw) * segw
        stack.append(a1 + " ")


def pop():
    """Pops an indentation string from the stack."""
    stack = _get_indent_stack()
    if stack:
        stack.pop()

class IndentContext:
    """Context manager for indentation that can be used with 'with' statements."""

    def __init__(self, a1=1, a2=None, log_func=None):
        self.a1 = a1
        self.a2 = a2
        self.log_func = log_func or logger_main.info

    def __enter__(self):
        push(self.a1, self.a2, log_func=self.log_func)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pop()
        return False

# --- Additional Utilities ---

def ellipse(text: str, max_length: int = 50) -> str:
    """
    Truncates a string to a maximum length and adds an ellipsis if it's too long.
    """
    if len(text) > max_length:
        return text[:max_length] + '...'
    return text

# RICH HANDLER
# ----------------------------------------


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


class CustomRichHandler(RichHandler):
    """
    A custom RichHandler that formats log messages to include the
    class and function name of the caller.
    """

    def __init__(self, *kargs, print_path, path_width=10, highlight, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.print_path = print_path
        self.highlight = highlight

        import json
        self.persistence_file = ".persistence/logging.json"

        saved_path_width = path_width
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, 'r') as f:
                    data = json.load(f)
                    saved_path_width = data.get('max_path_width', path_width)
            except (json.JSONDecodeError, IOError):
                # If file is malformed or cannot be read, use default.
                pass

        self.path_width = saved_path_width


    def render_message(self, record: logging.LogRecord, message: str):
        """
        Renders the log message with a prefix containing the caller's context.
        """

        def _align_multiline(txt, left_, extra=0):
            # Push everything after the first line with left width
            lines = txt.splitlines()
            if len(lines) > 1:
                txt = ""
                if lines[0].strip():
                    txt = lines[0] + "\n"
                else:
                    lines = lines[1:]
                newlines = list([" " * (len(left_) + extra) + line for line in lines[1:]])
                txt += "\n".join(newlines)
            return txt

        # --- PATH ---
        path = ""
        cls = getattr(record, "classname", None)
        if cls:
            path = f"({cls}.{record.funcName})"
        else:
            filename = os.path.basename(record.pathname).replace(".py", "")
            path = f"({filename}.{record.funcName})"

        path = f"{path}: "
        path_w = len(path)
        path = path.rjust(self.path_width + 2)  # + colon space

        idt = "".join(_get_indent_stack())

        left = f"{path}{idt}"
        text = PrintedText(record.msg or "", highlight=self.highlight).markup  # Bake any rich object, since we need to know it's gonna be multiline
        # print(f"ALIGN TO {len(left)} (lines: {len(text.splitlines())})")

        # print("BEFORE:")
        # print(text)
        text_aligned = _align_multiline(text, left)
        # print("AFTER:")
        # print(text)
        text = PrintedText(text, highlight=False).markup

        # if self.print_path:
        full = f"[dim]{left}[/dim]{text_aligned}"

        # PERSISTENCE
        # ----------------------------------------
        if path_w > self.path_width:
            self.path_width = path_w
            os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)
            try:
                with open(self.persistence_file, 'w') as f:
                    import json
                    json.dump({'max_path_width': self.path_width}, f)
                    logger.info(f"Wrote max_pad_width={self.path_width} to persistence file {self.persistence_file}")
            except IOError:
                pass  # Don't crash on logging persistence error

        return Text.from_markup(full)
