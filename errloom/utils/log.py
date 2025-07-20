import logging
import os
import sys
from datetime import datetime
from functools import wraps
from math import ceil
from pathlib import Path
from threading import local
from typing import Callable, Optional, Any
from contextlib import contextmanager
from time import perf_counter
import colorsys

import rich.traceback
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

logger = logging.getLogger(__name__)

# TRACER FUNCTIONALITY
# ----------------------------------------

# Global tracer state
trace_indent = 0
print_trace = False

def set_trace_enabled(enabled: bool):
    """Enable or disable tracing globally."""
    global print_trace
    print_trace = enabled

def is_trace_enabled() -> bool:
    """Check if tracing is currently enabled."""
    global print_trace
    return print_trace

def get_color_for_time(seconds: float) -> str:
    """Get a Rich color style based on elapsed time for visual feedback."""
    from rich.style import Style
    
    # Define our gradient keypoints (time in seconds, hue)
    keypoints = [
        (0, 120),  # Green (very fast)
        (0.5, 80),  # Yellow-Green
        (1, 60),  # Yellow
        (1.5, 30),  # Orange
        (2, 0),  # Red (very slow)
    ]

    # Find the appropriate segment of the gradient
    for i, (time, hue) in enumerate(keypoints):
        if seconds <= time:
            if i == 0:
                r, g, b = colorsys.hsv_to_rgb(hue / 360, 1, 1)
                return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"

            # Interpolate between this keypoint and the previous one
            prev_time, prev_hue = keypoints[i - 1]
            t = (seconds - prev_time) / (time - prev_time)
            interpolated_hue = prev_hue + t * (hue - prev_hue)

            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(interpolated_hue / 360, 1, 1)
            return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"

    # If we're past the last keypoint, return the color for the last keypoint
    r, g, b = colorsys.hsv_to_rgb(keypoints[-1][1] / 360, 1, 1)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"

@contextmanager
def tracer(name: str, threshold: float | int = 0.0, *, force=False, show_args: bool = False, args: Optional[tuple] = None, kwargs: Optional[dict] = None):
    """
    Context manager for timing code execution with optional threshold filtering.
    
    Args:
        name: Name of the operation being traced
        threshold: Minimum time threshold to print (in seconds)
        force: Force printing regardless of global trace setting
        show_args: Whether to show function arguments in the trace
        args: Function arguments to display
        kwargs: Function keyword arguments to display
    """
    global trace_indent
    
    if not print_trace and not force and threshold < 0.002:
        yield
        return

    start_time = perf_counter()
    trace_indent += 1
    
    # Build the display name with arguments if requested
    display_name = name
    if show_args and (args or kwargs):
        arg_strs = []
        if args:
            arg_strs.extend([value_to_print_str(arg) for arg in args])
        if kwargs:
            arg_strs.extend([f"{k}={value_to_print_str(v)}" for k, v in kwargs.items()])
        if arg_strs:
            display_name = f"{name}({', '.join(arg_strs)})"
    
    try:
        yield lambda: perf_counter() - start_time
    finally:
        trace_indent -= 1
        elapsed_time = perf_counter() - start_time
        
        if threshold is not None and elapsed_time < threshold:
            return

        if elapsed_time >= 0.002:  # Only print if time is 2ms or more
            color_style = get_color_for_time(elapsed_time)
            # Use the existing logging system - RichHandler will handle indentation
            logger_main.info(f"[{color_style}]({elapsed_time:.3f}s) {display_name}[/{color_style}]")

def _trace_wrapper(func: Callable, print_args: bool, *args: Any, **kwargs: Any) -> Any:
    """Internal wrapper for function tracing."""
    with tracer(func.__name__, show_args=print_args, args=args if print_args else None, kwargs=kwargs if print_args else None) as timer:
        return func(*args, **kwargs)

def trace_decorator(func: Callable) -> Callable:
    """Decorator to trace function execution with arguments."""
    def trace_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _trace_wrapper(func, True, *args, **kwargs)
    return trace_wrapper

def trace_decorator_noargs(func: Callable) -> Callable:
    """Decorator to trace function execution without arguments."""
    def trace_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _trace_wrapper(func, False, *args, **kwargs)
    return trace_wrapper

def trace_function(name: Optional[str] = None, threshold: float = 0.0, *, force=False):
    """
    Decorator factory for tracing functions with custom name and threshold.
    
    Args:
        name: Custom name for the trace (defaults to function name)
        threshold: Minimum time threshold to print
        force: Force printing regardless of global trace setting
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            trace_name = name or func.__name__
            with tracer(trace_name, threshold=threshold, force=force, show_args=True, args=args, kwargs=kwargs) as timer:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def value_to_print_str(v):
    """Convert a value to a printable string representation."""
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return f"ndarray({v.shape})"
    except ImportError:
        pass

    try:
        from PIL import Image
        if isinstance(v, Image.Image):
            return f"PIL({v.width}x{v.height}, {v.mode})"
    except ImportError:
        pass

    # Handle numeric types
    if isinstance(v, (float, complex)) and not isinstance(v, bool):
        return f"{v:.2f}"

    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v}"

    # Handle tuples/lists of floats
    if isinstance(v, (tuple, list)) and len(v) > 0 and isinstance(v[0], float):
        v = list(f"{x:.2f}" for x in v)
        ret = "(" + "  ".join(v) + ")"
        return ret

    # Limit floats to 2 decimals
    if isinstance(v, float):
        return f"{v:.2f}"

    # Handle PyTorch tensors
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return f"Tensor({v.shape})"
    except ImportError:
        pass

    return f"{v}"

# ENHANCED LOGGER CLASS
# ----------------------------------------

class EnhancedLogger(logging.Logger):
    """Logger with enhanced push/pop indent functionality."""

    def push(self, a1: int | str = 1, a2: Optional[str] = None, log_func: Optional[Callable] = None) -> None:
        """Push indentation with optional logging function."""
        return push(a1, a2, log_func=log_func or self.info)

    def pop(self) -> None:
        """Pop indentation from the stack."""
        return pop()

    def indent_ctx(self, a1: int | str = 1, a2: Optional[str] = None, log_func: Optional[Callable] = None) -> 'IndentContext':
        """Context manager for temporary indentation."""
        return IndentContext(a1, a2, log_func=log_func or self.info)

    def push_debug(self, a1: int | str = 1, a2: Optional[str] = None) -> None:
        """Push with debug level logging."""
        return push(a1, a2, log_func=self.debug)

    def push_info(self, a1: int | str = 1, a2: Optional[str] = None) -> None:
        """Push with info level logging."""
        return push(a1, a2, log_func=self.info)

    def push_warning(self, a1: int | str = 1, a2: Optional[str] = None) -> None:
        """Push with warning level logging."""
        return push(a1, a2, log_func=self.warning)

    def push_error(self, a1: int | str = 1, a2: Optional[str] = None) -> None:
        """Push with error level logging."""
        return push(a1, a2, log_func=self.error)

    def push_critical(self, a1: int | str = 1, a2: Optional[str] = None) -> None:
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


def generate_log_filename(script_name: Optional[str] = None) -> str:
    """
    Generate a log filename based on current time and main script name.

    Args:
        script_name: Optional script name. If None, auto-detects from sys.argv[0]

    Returns:
        Path to log file in logs/ directory
    """
    if script_name is None:
        # Get the main script name from sys.argv[0]
        script_path = sys.argv[0] if sys.argv else "unknown"
        script_name = Path(script_path).stem

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create filename
    filename = f"{timestamp}_{script_name}.log"

    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    return str(logs_dir / filename)


def setup_logging(
    level: str = "DEBUG",
    print_path: bool = True,
    highlight: bool = True,
    enable_file_logging: bool = True,
    log_filename: Optional[str] = None,
    persistence_file: Optional[str] = None
) -> str:
    """
    Setup basic logging configuration for the errloom package.

    Args:
        level: The logging level to use. Defaults to "DEBUG".
        print_path: Whether to print file paths in console output.
        highlight: Whether to enable syntax highlighting in console output.
        enable_file_logging: Whether to enable logging to file.
        log_filename: Optional custom log filename. If None, auto-generates one.
        persistence_file: Optional path for logging persistence file. If None, uses default.

    Returns:
        Path to the log file if file logging is enabled, empty string otherwise.
    """

    # Get the root logger for the package
    logger = logging.getLogger()
    logger.setLevel(level.upper())

    # if logger.hasHandlers():
    #     logger.handlers.clear()

    # Create a RichHandler for console output
    if not any(isinstance(handler, CustomRichHandler) for handler in logger.handlers):
        handler = CustomRichHandler(
            rich_tracebacks=True, 
            show_path=False, 
            console=cl, 
            markup=True, 
            print_path=print_path, 
            highlight=highlight,
            persistence_file=persistence_file
        )
        logger.addHandler(handler)

    # Add file handler if enabled
    log_file_path = ""
    if enable_file_logging:
        if log_filename is None:
            log_file_path = generate_log_filename()
        else:
            log_file_path = log_filename
            # Ensure parent directory exists
            Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

        # Create file handler with a simple formatter (no rich formatting for file)
        if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            file_handler.setLevel(level.upper())

            # Create a simple formatter for file output (no colors/markup)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    # Prevent the logger from propagating messages to the root logger
    # logger.propagate = False
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)

    # Suppress noisy debug messages from various modules
    logging.getLogger("spec").setLevel(logging.WARNING)
    logging.getLogger("spec.read").setLevel(logging.WARNING)
    logging.getLogger("selector_events").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Suppress other common noisy loggers
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("concurrent.futures").setLevel(logging.WARNING)

    # suppress annoying insignificant bullshit spam-and-harrass-by-default behavior
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)

    # Add a custom filter to suppress specific noisy messages
    # This filter removes repetitive debug messages from third-party libraries
    # (HuggingFace datasets, fsspec, asyncio) that clutter application logs
    # with low-level file system operations and internal library details
    class NoiseFilter(logging.Filter):
        def filter(self, record):
            # Suppress specific noisy debug messages from third-party libraries
            if record.levelno == logging.DEBUG:
                message = record.getMessage() if hasattr(record, 'getMessage') else str(record.msg)
                if any(noise in message for noise in [
                    'HfFileSystem',          # HuggingFace filesystem operations
                    'read: ',                # File read operations
                    'readahead:',           # Cache readahead operations
                    'hits, ',               # Cache hit statistics
                    'misses,',              # Cache miss statistics
                    'total requested bytes', # Byte count summaries
                    'Using selector:',      # Asyncio selector messages
                ]):
                    return False
            return True

    # Apply the filter to all handlers
    for handler in logger.handlers:
        handler.addFilter(NoiseFilter())

    return log_file_path

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

    return logger  # type: ignore



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

# TODO support a 'spacer' argument to push which is the space by default. It's saved and deferred for the next push, and prepended instead of appended. The final space is instead part of our RichHandler. This allows doing sub-indents for sections of a function, like WARE.init, WARE.main, WARE.cleanup

_indent_stack = []

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


def push(a1: int | str = 1, a2: Optional[str] = None, log_func: Optional[Callable] = None) -> None:
    """Pushes an indentation string onto the stack."""
    if log_func is None:
        log_func = logger_main.info

    stack = _indent_stack
    seg = ".. "
    segw = len(seg)
    if isinstance(a1, int):
        stack.append(seg * a1)
    elif isinstance(a1, str) and a2 is not None:
        log_func(f"{a1} {a2}")
        # Push spaces to align subsequent logs under the message
        stack.append(a1 + " ")
    elif isinstance(a1, str):
        # log_func(a1)
        i = ceil(len(a1) / segw) * segw
        stack.append(a1 + " ")


def pop():
    """Pops an indentation string from the stack."""
    stack = _indent_stack
    if stack:
        stack.pop()

class IndentContext:
    """Context manager for indentation that can be used with 'with' statements."""

    def __init__(self, a1: int | str = 1, a2: Optional[str] = None, log_func: Optional[Callable] = None):
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

    def __init__(self, *kargs, print_path, path_width=10, highlight, persistence_file=None, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.print_path = print_path
        self.highlight = highlight

        import json
        self.persistence_file = persistence_file or ".persistence/logging.json"

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

        idt = "".join(_indent_stack)

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
