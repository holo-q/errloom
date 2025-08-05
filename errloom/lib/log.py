import colorsys
import concurrent.futures
import logging
import os
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional

import rich.traceback
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

from errloom import paths  # noqa: E402
# Centralized persistence paths
# Storage is available if needed for future extensions; avoid adding dynamic fields
# Persisted storage (lives under ~/.errloom via paths.userdir)
from errloom.storage import application as storage_application  # noqa: E402

# Centralized persistence paths (no dynamic writes to Storage schema)
# Centralized persistence

logger = logging.getLogger(__name__)

# Safe terminal width detection for non-TTY environments (tests / CI)
def _safe_terminal_width(default_width: int = 120) -> int:
    try:
        if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
            return os.get_terminal_size(sys.stdout.fileno()).columns
    except (OSError, ValueError):
        pass
    try:
        return os.get_terminal_size().columns
    except (OSError, ValueError):
        return default_width

TERM_WIDTH = _safe_terminal_width()

# Normalize any prior width derivations to TERM_WIDTH (single integer)
# If any previous code attempted tuple-unpacking from .columns, fix it:
# Before (invalid): width, _ = os.get_terminal_size(sys.stdout.fileno()).columns
# After (valid):   width = TERM_WIDTH
try:
    # Shadow a common name if used elsewhere
    WIDTH = TERM_WIDTH
except Exception:
    pass

# Safely set `width` used by formatting; prefer not to overwrite if already defined correctly.
try:
    # If `width` is not defined or was incorrectly set via tuple-unpack on int, ensure it's an int.
    if "width" not in globals() or not isinstance(globals().get("width"), int):
        width = TERM_WIDTH
except Exception:
    width = TERM_WIDTH

cl = Console(width=TERM_WIDTH)

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



# TRACER FUNCTIONALITY
# ----------------------------------------

# Global tracer state
print_trace = False

def set_trace_enabled(enabled: bool):
    """Enable or disable tracing globally."""
    global print_trace
    print_trace = enabled

def is_trace_enabled() -> bool:
    """Check if tracing is currently enabled."""
    global print_trace
    return print_trace

def _get_trace_indent():
    """Get the thread-local trace indent, creating it if it doesn't exist."""
    if not hasattr(_thread_local, 'trace_indent'):
        _thread_local.trace_indent = 0
    return _thread_local.trace_indent

def _set_trace_indent(value):
    """Set the thread-local trace indent."""
    _thread_local.trace_indent = value

def get_color_for_time(seconds: float) -> str:
    """Get a Rich color style based on elapsed time for visual feedback."""

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
    if not print_trace and not force and threshold < 0.002:
        yield
        return

    start_time = perf_counter()
    current_indent = _get_trace_indent()
    _set_trace_indent(current_indent + 1)

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
        _set_trace_indent(current_indent)
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

    def get_current_context(self) -> list:
        """Get a copy of the current thread's indent stack for passing to child threads."""
        return get_current_context()

    def set_context(self, context: list) -> None:
        """Set the thread-local indent stack to a specific context."""
        set_indent_stack(context)

    # --- Convenience helpers ---
    def info_hl(self, msg: Any, hl: bool = True, stacklevel: int = 1) -> None:
        """info with per-record highlight override."""
        self.info(msg, extra={"highlight": hl}, stacklevel=stacklevel + 1)

    def debug_hl(self, msg: Any, hl: bool = True, stacklevel: int = 1) -> None:
        """debug with per-record highlight override."""
        self.debug(msg, extra={"highlight": hl}, stacklevel=stacklevel + 1)

    def warning_hl(self, msg: Any, hl: bool = True, stacklevel: int = 1) -> None:
        """warning with per-record highlight override."""
        self.warning(msg, extra={"highlight": hl}, stacklevel=stacklevel + 1)

    def error_hl(self, msg: Any, hl: bool = True, stacklevel: int = 1) -> None:
        """error with per-record highlight override."""
        self.error(msg, extra={"highlight": hl}, stacklevel=stacklevel + 1)

    def critical_hl(self, msg: Any, hl: bool = True, stacklevel: int = 1) -> None:
        """critical with per-record highlight override."""
        self.critical(msg, extra={"highlight": hl}, stacklevel=stacklevel + 1)

    def with_extra(self, level: int, msg: Any, extra: dict, stacklevel: int = 1) -> None:
        """Log at arbitrary level with custom extra fields, preserving stacklevel contract."""
        self.log(level, msg, extra=extra, stacklevel=stacklevel + 1)

def generate_log_filename(script_name: Optional[str] = None) -> str:
    """
    Generate a log filename based on current time and main script name.

    Returns a path under ~/.errloom/logs to respect centralized persistence.
    """
    if script_name is None:
        script_path = sys.argv[0] if sys.argv else "unknown"
        script_name = Path(script_path).stem

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{script_name}.log"

    # Persist logs under ~/.errloom/logs
    logs_dir = paths.user_logs
    logs_dir.mkdir(parents=True, exist_ok=True)

    return str(logs_dir / filename)


def setup_logging(
    level: str = "DEBUG",
    print_paths: bool = True,
    highlight: bool = True,
    enable_file_logging: bool = True,
    log_filename: Optional[str] = None,
    print_threads: bool = False,
    print_time: bool = False,
    reset_log_columns: bool = False
) -> str:
    """
    Setup basic logging configuration for the errloom package.

    Args:
        level: The logging level to use. Defaults to "DEBUG".
        print_paths: Whether to print file paths in console output.
        highlight: Whether to enable syntax highlighting in console output.
        enable_file_logging: Whether to enable logging to file.
        log_filename: Optional custom log filename. If None, auto-generates one.
        persistence_file: Optional path for logging persistence file. If None, uses default.
        print_threads: Whether to print thread names in console output.
        print_time: Whether to print timestamps in console output. Defaults to False.
        reset_log_columns: Whether to reset the persisted column width.

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
            show_time=print_time,
            console=cl,
            markup=True,
            print_paths=print_paths,
            highlight=highlight,
            print_thread_name=print_threads,
            reset_columns=reset_log_columns
        )
        logger.addHandler(handler)

    # Add file handler if enabled
    log_file_path = ""
    if enable_file_logging:
        if log_filename is None:
            log_file_path = generate_log_filename()
        else:
            # If a custom file is provided, still ensure it's under userdir when relative
            p = Path(log_filename)
            if not p.is_absolute():
                p = paths.user_logs / p
            log_file_path = p.as_posix()
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

def disable_logger(name: Optional[str] = None) -> None:
    """
    Disable a logger and silence all of its handlers.
    If name is None, disables the root logger.

    This is useful in tests or focused runs to mute specific subsystems
    like 'errloom.tapestry' without affecting the rest of the logging.

    Args:
        name: The logger name to disable. Defaults to root if None.
    """
    logger = getLogger(name) if name else logging.getLogger()
    logger.disabled = True
    for handler in list(logger.handlers):
        handler.setLevel(logging.CRITICAL + 1)

def save_session_width_to_persistence():
    """Save the current session's max width +10% to persistence."""
    logger = logging.getLogger()
    for handler in logger.handlers:
        if isinstance(handler, CustomRichHandler):
            handler.save_session_width_to_persistence()
            break

def logging_level_from_str(level_str: str, default: int = logging.INFO) -> int:
    s = str(level_str or "").strip().lower()
    mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "warn": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
        "fatal": logging.CRITICAL,
    }
    return mapping.get(s, default)


def _iter_known_logger_names() -> list[str]:
    """
    Back-compat: iterate currently-registered loggers.
    Kept for runtime-added names, but no longer the primary discovery mechanism.
    """
    names = set(["main"])
    try:
        names.update(logging.Logger.manager.loggerDict.keys())
    except Exception:
        pass
    return sorted(n for n in names if isinstance(n, str) and n != "")


def get_known_loggers_by_short_name(restrict_prefix: str = "errloom") -> dict[str, list[str]]:
    """
    Build mapping from final component ('tapestry') to list of full logger names.

    New strategy:
    - Predict module paths by scanning the project package directory tree (filesystem),
      not by relying on logging registry (which only contains already-imported modules).
    - Restrict to our package or 'main' to avoid 3rd-party loggers.
    """
    out: dict[str, list[str]] = {}

    # Always include 'main' short mapping
    out.setdefault("main", []).append("main")

    try:
        # Locate installed package root path
        import importlib
        pkg = importlib.import_module(restrict_prefix)
        pkg_path = getattr(pkg, "__path__", None)
        roots = list(pkg_path) if pkg_path is not None else []
    except Exception:
        roots = []

    def add_module_modname(modname: str):
        if not modname.startswith(f"{restrict_prefix}."):
            return
        short = modname.split(".")[-1]
        out.setdefault(short, []).append(modname)

    # Walk all package roots to discover python modules
    for root in roots:
        try:
            root_path = Path(root)
            if not root_path.exists():
                continue
            for p in root_path.rglob("*.py"):
                # Skip private/dunder cache
                if p.name.startswith("_") and p.name != "__init__.py":
                    continue
                    
                # Build relative path from package root
                try:
                    rel = p.relative_to(root_path)
                except ValueError:
                    # Shouldn't happen since we walked from root_path, but safety
                    continue

                # Build module name
                parts = [restrict_prefix]
                
                # Handle __init__.py - module is the package directory
                if p.name == "__init__.py":
                    parts.extend([x for x in rel.parent.parts if x])
                else:
                    # Regular .py file - module is the file path without extension
                    parts.extend([x for x in rel.with_suffix("").parts if x])
                
                modname = ".".join(parts)
                add_module_modname(modname)
        except Exception:
            # Fail-safe: continue with whatever we built so far
            continue

    # Also include any already-registered matching names, without creating a hard dependency
    try:
        for full in _iter_known_logger_names():
            if full == "main" or full.startswith(f"{restrict_prefix}."):
                short = full.split(".")[-1]
                out.setdefault(short, []).append(full)
    except Exception:
        pass

    # Deduplicate lists while preserving order
    for k, v in list(out.items()):
        seen = set()
        dedup = []
        for n in v:
            if n not in seen:
                seen.add(n)
                dedup.append(n)
        out[k] = dedup

    return out


def resolve_logger_targets(tokens: list[str]) -> list[str]:
    """
    Resolve a list of module tokens into full logger names.

    Resolution order:
    1) If token matches a predicted fully-qualified module name (from filesystem scan), use it.
    2) If token matches a short name, expand to all predicted modules with that short name.
    3) Else, accept token as-is so the logger can be created on demand.
    """
    by_short = get_known_loggers_by_short_name()
    predicted_full = set(n for lst in by_short.values() for n in lst)

    result: list[str] = []
    for t in tokens:
        t = (t or "").strip()
        if not t:
            continue
        # Exact full-name match in predicted set
        if t in predicted_full:
            result.append(t)
            continue
        # Short-name expansion
        if t in by_short:
            result.extend(by_short[t])
            continue
        # Fallback: treat as fqname for forward-compat (logger will be created lazily)
        result.append(t)

    # Deduplicate preserving order
    seen = set()
    deduped = []
    for n in result:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped


def parse_logs_arg(expr: str, default_level: int) -> dict[str, int]:
    """
    Parse --logs expression: 'tapestry:debug,foo,bar:error'
    Returns mapping short/full tokens -> level int. Levels not provided use default_level.
    """
    out: dict[str, int] = {}
    if not expr:
        return out
    parts = [p.strip() for p in expr.split(",") if p.strip()]
    for p in parts:
        if ":" in p:
            name, lvl = p.split(":", 1)
            level_int = logging_level_from_str(lvl, default_level)
            out[name.strip()] = level_int
        else:
            out[p] = default_level
    return out


def apply_logger_levels(mapping: dict[str, int]) -> None:
    """
    Apply levels to all resolved loggers according to mapping keyed by token (short or fqname).
    """
    if not mapping:
        return
    # Merge tokens into actual names
    expanded: dict[str, int] = {}
    for token, level in mapping.items():
        targets = resolve_logger_targets([token])
        for name in targets:
            expanded[name] = level
    for name, level in expanded.items():
        try:
            logging.getLogger(name).setLevel(level)
        except Exception:
            pass


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

# COLORIZE FUNCTIONS
# ----------------------------------------

# Color scheme mapping for consistent theming across the application
_COLOR_SCHEME = {
    # Core entities
    'session': '[white]',
    'target': '[bright_green]',
    'model': '[white]',
    'client': '[bright_magenta]',

    # Status indicators
    'error': '[bold red]',
    'success': '[bold green]',
    'warning': '[yellow]',
    'info': '[cyan]',
    'debug': '[dim]',

    # UI elements
    'title': '[bold bright_blue]',
    'path': '[bright_cyan]',
    'field_label': '[bold cyan]',
    'rule_title': '[bold cyan]',

    # Mode indicators
    'mode_dry': '[yellow]',
    'mode_production': '[green]',
    'mode_dump': '[blue]',

    # Progress and data
    'completion': '[bold green]',
    'progress': '[magenta]',
    'data': '[bright_yellow]',
    'deployment': '[bold cyan]',

    # Holoware-specific
    'loom': '[bright_blue]',
    'holoware': '[bright_green]',
    'session_name': '[white]',
    'placeholder': '[bright_magenta]',
    'command_name': '[bright_cyan]',

    # Symbols
    'check_mark': '[green]✓[/]',
    'cross_mark': '[red]✗[/]',
    'arrow': '[green]→[/]',
}

def _colorize(text: str, color_key: str) -> str:
    """Apply color formatting to text using the color scheme."""
    color = _COLOR_SCHEME.get(color_key, '[white]')
    if color.endswith('[/]'):
        return color  # Symbol colors already include closing tag
    return f"{color}{text}[/]"

def colorize_session(text: str) -> str:
    """Colorize session names and identifiers."""
    return _colorize(text, 'session')

def colorize_target(text: str) -> str:
    """Colorize target names (holoware, loom names)."""
    return _colorize(text, 'target')

def colorize_model(text: str) -> str:
    """Colorize model names."""
    return _colorize(text, 'model')

def colorize_client(text: str) -> str:
    """Colorize client type names."""
    return _colorize(text, 'client')

def colorize_error(text: str) -> str:
    """Colorize error messages."""
    return _colorize(text, 'error')

def colorize_success(text: str) -> str:
    """Colorize success messages."""
    return _colorize(text, 'success')

def colorize_warning(text: str) -> str:
    """Colorize warning messages."""
    return _colorize(text, 'warning')

def colorize_info(text: str) -> str:
    """Colorize informational messages."""
    return _colorize(text, 'info')

def colorize_debug(text: str) -> str:
    """Colorize debug messages."""
    return _colorize(text, 'debug')

def colorize_title(text: str) -> str:
    """Colorize titles and headers."""
    return _colorize(text, 'title')

def colorize_path(text: str) -> str:
    """Colorize file paths and locations."""
    return _colorize(text, 'path')

def colorize_mode_dry(text: str) -> str:
    """Colorize dry mode messages."""
    return _colorize(text, 'mode_dry')

def colorize_mode_production(text: str) -> str:
    """Colorize production mode messages."""
    return _colorize(text, 'mode_production')

def colorize_mode_dump(text: str) -> str:
    """Colorize dump mode messages."""
    return _colorize(text, 'mode_dump')

def colorize_field_label(text: str) -> str:
    """Colorize field labels (like 'Session:', 'Model:')."""
    return _colorize(text, 'field_label')

def colorize_completion(text: str) -> str:
    """Colorize completion/training completion messages."""
    return _colorize(text, 'completion')

def colorize_progress(text: str) -> str:
    """Colorize progress indicators."""
    return _colorize(text, 'progress')

def colorize_data(text: str) -> str:
    """Colorize data-related information."""
    return _colorize(text, 'data')

def colorize_rule_title(text: str) -> str:
    """Colorize rule titles for visual separation."""
    return _colorize(text, 'rule_title')

def colorize_deployment(text: str) -> str:
    """Colorize deployment-related messages."""
    return _colorize(text, 'deployment')

def colorize_loom(text: str) -> str:
    """Colorize loom class names and references."""
    return _colorize(text, 'loom')

def colorize_holoware(text: str) -> str:
    """Colorize holoware file names and references."""
    return _colorize(text, 'holoware')

def colorize_session_name(text: str) -> str:
    """Colorize session names and identifiers in placeholders."""
    return _colorize(text, 'session_name')

def colorize_placeholder(text: str) -> str:
    """Colorize generic placeholders."""
    return _colorize(text, 'placeholder')

def colorize_command_name(text: str) -> str:
    """Colorize command names in placeholders."""
    return _colorize(text, 'command_name')

def colorize_check_mark() -> str:
    """Standard green checkmark."""
    return _COLOR_SCHEME['check_mark']

def colorize_cross_mark() -> str:
    """Standard red cross mark."""
    return _COLOR_SCHEME['cross_mark']

def colorize_arrow() -> str:
    """Standard arrow for file operations."""
    return _COLOR_SCHEME['arrow']

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
    logger.info(colorize_info(text), stacklevel=stacklevel + 1)
    logger.info("")

def log_stacktrace_to_file_only(logger, exception: Exception, context: str = ""):
    """
    Log a full stacktrace to file only, not to console.
    This is useful for debugging while keeping console output clean.

    Args:
        logger: Logger instance to use
        exception: The exception that occurred
        context: Optional context string to include in the log
    """
    import traceback
    from datetime import datetime

    # Get the full stacktrace
    stacktrace = traceback.format_exc()

    # Create a detailed error message
    error_msg = f"EXCEPTION{': ' + context if context else ''}\n"
    error_msg += f"Exception: {type(exception).__name__}: {str(exception)}\n"
    error_msg += f"Stacktrace:\n{stacktrace}"

    # Find file handlers from the original logger and write directly to them
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                # Write directly to the file with timestamp
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_entry = f"{timestamp} - {logger.name} - ERROR - {error_msg}\n"
                handler.stream.write(log_entry)
                handler.stream.flush()
            except Exception as e:
                # If direct write fails, fall back to logging (but this might go to console)
                logger.error(f"Failed to write stacktrace to file: {e}")
                break

class LogContext:
    """Context manager for wrapping code blocks with intro/outro logging."""

    def __init__(self, intro_msg: str, outro_msg: Optional[str] = None, logger=logger_main):
        self.intro = intro_msg
        self.outro = outro_msg or intro_msg.replace("Starting", "Completed").replace(
            "Initializing", "Initialized"
        )
        self.logger = logger

    def __enter__(self):
        log(colorize_info(self.intro), logger=self.logger, stacklevel=2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            logl(f"{colorize_check_mark()} {self.outro}", logger=self.logger, stacklevel=2)
        else:
            logl(f"{colorize_cross_mark()} {self.outro} failed: {exc_val}", logger=self.logger, stacklevel=2)
            return False

# GLOBAL INDENT TRACKER - thread-local to handle concurrent logging
# ----------------------------------------

# TODO support a 'spacer' argument to push which is the space by default. It's saved and deferred for the next push, and prepended instead of appended. The final space is instead part of our RichHandler. This allows doing sub-indents for sections of a function, like WARE.init, WARE.main, WARE.cleanup

# Thread-local storage for indent stacks to prevent conflicts in multi-threaded environments
_thread_local = threading.local()

def _get_indent_stack():
    """Get the thread-local indent stack, creating it if it doesn't exist."""
    if not hasattr(_thread_local, 'indent_stack'):
        # Start fresh - context should be passed explicitly
        _thread_local.indent_stack = []

    return _thread_local.indent_stack

def set_indent_stack(context):
    """Set the thread-local indent stack to a specific context."""
    _thread_local.indent_stack = context.copy()

def get_current_context():
    """Get a copy of the current thread's indent stack for passing to child threads."""
    return _get_indent_stack().copy()

def clear_stack():
    _get_indent_stack().clear()

class ContextAwareThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """
    A ThreadPoolExecutor that automatically propagates the errloom logging indent context
    from the submitting thread to the worker thread.
    """
    def submit(self, fn, /, *args, **kwargs):
        """
        Submits a callable to be executed, wrapping it to carry over the logging context.
        """
        # This is called in the parent thread, so we capture its context.
        parent_context = get_current_context()

        def fn_with_context(*args, **kwargs):
            # This runs in the worker thread. We set the captured context.
            set_indent_stack(parent_context)
            try:
                return fn(*args, **kwargs)
            finally:
                # Context will be overwritten by the next task, so no cleanup is needed.
                pass

        # Submit the wrapped function to the actual executor.
        return super().submit(fn_with_context, *args, **kwargs)

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
    """Pushes an indentation string onto the thread-local stack."""
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
        # log_func(a1)
        i = ceil(len(a1) / segw) * segw
        stack.append(a1 + " ")


def pop():
    """Pops an indentation string from the thread-local stack."""
    stack = _get_indent_stack()
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

def to_json_text(obj, *, indent: int = 4, ensure_ascii: bool = False, max_depth: int = 6, redactions: list[str] | tuple[str, ...] = ()) -> str:
    """
    Convert an arbitrary Python object into a JSON-formatted string suitable for logs.

    - Uses double quotes and proper escaping
    - Supports dataclasses and simple objects by walking __dict__
    - Limits recursion depth and detects cycles
    - Supports attribute blacklist via dot-paths (e.g., "foo", "foo.bar")
    - Falls back to str(obj) if serialization fails
    """
    import json
    try:
        from dataclasses import asdict, is_dataclass  # type: ignore
    except Exception:
        def is_dataclass(_: object) -> bool:  # type: ignore
            return False
        def asdict(x: object):  # type: ignore
            return x

    PRIMITIVES = (str, int, float, bool, type(None), bytes)

    # Normalize blacklist into a set of dot-paths for O(1) checks
    bl_set: set[str] = set(redactions or ())

    def _is_blacklisted(path: str) -> bool:
        # Exact path match
        if path in bl_set:
            return True
        # Any parent path with wildcard semantics would be nice, but keep explicit for now
        return False

    def _safe_str(x: object) -> str:
        try:
            return str(x)
        except Exception:
            return f"<unprintable {type(x).__name__}>"

    def to_jsonable(v: object, depth: int, seen: set[int], path: str = "") -> object:
        # Depth guard
        if depth > max_depth:
            return f"<max_depth {max_depth} reached: {type(v).__name__}>"

        # Primitives
        if isinstance(v, PRIMITIVES):
            if isinstance(v, bytes):
                # limit size for bytes
                return v.decode(errors="replace")[:2048]
            return v

        oid = id(v)
        if oid in seen:
            return f"<recursion {type(v).__name__} id={oid}>"
        seen.add(oid)

        # Dict-like
        if isinstance(v, dict):
            out = {}
            for k, x in v.items():
                # Ensure keys are strings
                try:
                    sk = k if isinstance(k, str) else _safe_str(k)
                except Exception:
                    sk = f"<bad_key {type(k).__name__}>"
                subpath = f"{path}.{sk}" if path else sk
                if _is_blacklisted(subpath):
                    out[sk] = "<redacted>"
                else:
                    out[sk] = to_jsonable(x, depth + 1, seen, subpath)
            return out

        # Sequences
        if isinstance(v, (list, tuple, set)):
            return [to_jsonable(x, depth + 1, seen, f"{path}[{i}]") for i, x in enumerate(v)]

        # Dataclass
        try:
            if is_dataclass(v):  # type: ignore[func-returns-value]
                return to_jsonable(asdict(v), depth + 1, seen, path)  # type: ignore[arg-type]
        except Exception:
            pass

        # Avoid descending into potentially huge/recursive frameworks
        mod = getattr(v, "__module__", "") or ""
        if mod.startswith("asyncio.") or mod.startswith("logging") or mod.startswith("rich"):
            return _safe_str(v)

        # Object with __dict__
        d = getattr(v, "__dict__", None)
        if isinstance(d, dict):
            try:
                out = {}
                for k, x in d.items():
                    sk = str(k)
                    subpath = f"{path}.{sk}" if path else sk
                    if _is_blacklisted(subpath):
                        out[sk] = f"<redacted:{subpath}>"
                    else:
                        out[sk] = to_jsonable(x, depth + 1, seen, subpath)
                return out
            except Exception:
                return _safe_str(v)

        # Fallback to str
        return _safe_str(v)

    try:
        payload = to_jsonable(obj, 0, set(), "")
        return json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent)
    except Exception:
        return _safe_str(obj)

# RICH HANDLER
# ----------------------------------------


def PrintedText(renderable, prefix_cols: int | None = None, width: int | None = None, highlight: bool = True) -> Text:
    """
    Render a rich renderable to Text using a width that accounts for the
    left prefix columns (path + indent) to minimize unintended wrapping.

    prefix_cols: total visible columns occupied by the left sidebar prefix.
                 This will be deducted from the console width so the payload
                 fits on the remaining line without wrapping.
    width:       optional total console width override (before prefix deduction).
    """
    # Conservative minimum for payload to avoid wrapping of banners/rules.
    MIN_PAYLOAD_WIDTH = 40

    # Determine the total console width baseline.
    total_width = width if isinstance(width, int) and width > 0 else TERM_WIDTH

    # Deduct the left prefix from the total, leaving at least MIN_PAYLOAD_WIDTH.
    left_cols = int(prefix_cols) if isinstance(prefix_cols, int) and prefix_cols > 0 else 0
    payload_width = max(MIN_PAYLOAD_WIDTH, total_width - left_cols - 12) # TODO why the fuck do we need to substract 10 ? none of the Rule(...) fit the console size without this.

    # Use a dedicated console at the computed payload width.
    c = Console(width=payload_width)
    with c.capture() as capture:
        c.print(renderable, highlight=highlight)
    return Text.from_ansi(capture.get())


class CustomRichHandler(RichHandler):
    """
    A custom RichHandler that formats log messages to include the
    class and function name of the caller.
    """

    def __init__(self, *kargs, print_paths, path_width=10, highlight, print_thread_name=False, reset_columns=False, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.highlight = highlight
        self.print_paths = print_paths
        self.print_threads = print_thread_name

        # Session tracking - track the longest width for this session
        self.session_max_width = path_width

        # Initialize from application storage (single source of truth)
        try:
            storage_application.read_json()
            saved_path_width = int(getattr(storage_application, "logging_max_path_width", path_width))
        except Exception:
            saved_path_width = path_width

        if reset_columns:
            saved_path_width = path_width
            try:
                storage_application.logging_max_path_width = int(path_width)
                storage_application.write_json()
            except Exception:
                pass

        self.path_width = saved_path_width

    def _save_persistence_data(self, data: dict):
        """Persist using AppStorage, no file-based persistence."""
        try:
            maxw = int(data.get('max_path_width', self.path_width))
            storage_application.logging_max_path_width = maxw
            storage_application.write_json()
        except Exception:
            pass

    def save_session_width_to_persistence(self):
        """Save the session's maximum width +10% using AppStorage."""
        if self.session_max_width > 0:
            padded_width = int(self.session_max_width * 1.1)
            self._save_persistence_data({'max_path_width': padded_width})
            logger.debug(f"Saved session max width {self.session_max_width} (+10% = {padded_width}) to storage")

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

        # Add thread name if enabled
        if self.print_threads:
            import threading
            current_thread = threading.current_thread()
            thread_name = current_thread.name
            # Show thread name for all threads when enabled, not just non-MainThread
            path = f"[{thread_name}] {path}"

        path = f"{path}: "
        path_w = len(path) - 6 - 3 # -6 for [bold] and 3 for [/]
        path = path.rjust(self.path_width + 2)  # + colon space

        idt = "".join(_get_indent_stack())

        if self.print_paths:
            left = f"{path}{idt}"
        else:
            left = f"{idt}"

        # Allow per-record highlight override via extra={"highlight": True|False}
        per_record_highlight = getattr(record, "highlight", None)
        highlight_flag = self.highlight if per_record_highlight is None else bool(per_record_highlight)

        # Compute visible columns used by the left sidebar (path + indent) as printed.
        left_cols = len(Text.from_markup(f"[dim]{left}[/dim]").plain)

        # Render once with sidebar width deducted to avoid unintended wrapping (esp. Rules/banners).
        rendered = PrintedText(
            record.msg or "",
            prefix_cols=left_cols,
            width=getattr(self.console, "width", TERM_WIDTH),
            highlight=highlight_flag
        ).markup

        # Align multiline payload under left prefix
        text_aligned = _align_multiline(rendered, left)

        # Final composed line with dimmed left sidebar + right payload
        full = f"[dim]{left}[/dim]{text_aligned}"

        # PERSISTENCE
        # ----------------------------------------
        # Track session max width separately from persistent width
        if path_w > self.session_max_width:
            self.session_max_width = path_w

        # Update persistent width for immediate use
        if path_w > self.path_width:
            self.path_width = path_w
            self._save_persistence_data({'max_path_width': self.path_width})
            logger.debug(f"Wrote max_pad_width={self.path_width} to AppStorage")

        return Text.from_markup(full)
