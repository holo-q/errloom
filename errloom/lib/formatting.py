def ellipse(text: str, max_length: int = 50) -> str:
    """
    Truncates a string to a maximum length and adds an ellipsis if it's too long.
    """
    if len(text) > max_length:
        return text[:max_length] + '...'
    return text

def ellipse_nl(text: str, max_length: int = 50) -> str:
    """
    Truncates a string to a maximum length and adds an ellipsis if it's too long.
    """
    return ellipse(text, max_length).replace("\n", "\\n")

_COLOR_SCHEME = {
    # Core entities
    'session':         '[white]',
    'target':          '[bright_green]',
    'model':           '[white]',
    'client':          '[bright_magenta]',

    # Status indicators
    'error':           '[bold red]',
    'success':         '[bold green]',
    'warning':         '[yellow]',
    'info':            '[cyan]',
    'debug':           '[dim]',

    # UI elements
    'title':           '[bold bright_blue]',
    'path':            '[bright_cyan]',
    'field_label':     '[bold cyan]',
    'rule_title':      '[bold cyan]',

    # Mode indicators
    'mode_dry':        '[yellow]',
    'mode_production': '[green]',
    'mode_dump':       '[blue]',

    # Progress and data
    'completion':      '[bold green]',
    'progress':        '[magenta]',
    'data':            '[bright_yellow]',
    'deployment':      '[bold cyan]',

    # Holoware-specific
    'loom':            '[bright_blue]',
    'holoware':        '[bright_green]',
    'session_name':    '[white]',
    'placeholder':     '[bright_magenta]',
    'command_name':    '[bright_cyan]',

    # Symbols
    'check_mark':      '[green]✓[/]',
    'cross_mark':      '[red]✗[/]',
    'arrow':           '[green]→[/]',

    # Context
    'frag': '[dim]'
}

def _colorize(text: str, color_key: str) -> str:
    """Apply color formatting to text using the color scheme."""
    color = _COLOR_SCHEME.get(color_key, '[white]')
    if color.endswith('[/]'):
        return color  # Symbol colors already include closing tag
    return f"{color}{text}[/]"


# context.py
def frag(text: str):
    return _colorize(ellipse_nl(text, 45), "frag")

def session(text: str) -> str: return _colorize(text, 'session')
def target(text: str) -> str: return _colorize(text, 'target')
def model(text: str) -> str: return _colorize(text, 'model')
def client(text: str) -> str: return _colorize(text, 'client')
def error(text: str) -> str: return _colorize(text, 'error')
def success(text: str) -> str: return _colorize(text, 'success')
def warning(text: str) -> str: return _colorize(text, 'warning')
def info(text: str) -> str: return _colorize(text, 'info')
def debug(text: str) -> str: return _colorize(text, 'debug')
def title(text: str) -> str: return _colorize(text, 'title')
def path(text: str) -> str: return _colorize(text, 'path')
def mode_dry(text: str) -> str: return _colorize(text, 'mode_dry')
def mode_production(text: str) -> str: return _colorize(text, 'mode_production')
def mode_dump(text: str) -> str: return _colorize(text, 'mode_dump')
def field_label(text: str) -> str: return _colorize(text, 'field_label')
def completion(text: str) -> str: return _colorize(text, 'completion')
def progress(text: str) -> str: return _colorize(text, 'progress')
def data(text: str) -> str: return _colorize(text, 'data')
def rule_title(text: str) -> str: return _colorize(text, 'rule_title')
def deployment(text: str) -> str: return _colorize(text, 'deployment')
def loom(text: str) -> str: return _colorize(text, 'loom')
def holoware(text: str) -> str: return _colorize(text, 'holoware')
def session_name(text: str) -> str: return _colorize(text, 'session_name')
def placeholder(text: str) -> str: return _colorize(text, 'placeholder')
def command_name(text: str) -> str: return _colorize(text, 'command_name')
def check_mark() -> str: return _COLOR_SCHEME['check_mark']
def cross_mark() -> str: return _COLOR_SCHEME['cross_mark']
def arrow() -> str: return _COLOR_SCHEME['arrow']
def command(command: str) -> str:
    """
    Colorize command examples with different colors for different parts.

    Args:
        command: The command string to colorize

    Returns:
        Rich markup string with appropriate colors applied
    """

    # Split command into parts
    parts = command.split()

    if not parts:
        return command

    colored_parts = []

    for i, part in enumerate(parts):
        if i == 0 and part == "uv":
            colored_parts.append(f"[bright_blue]{part}[/]")
        elif i == 1 and part == "run":
            colored_parts.append(f"[bright_blue]{part}[/]")
        elif i == 2 and part == "main":
            colored_parts.append(f"[cyan]{part}[/]")
        elif part.endswith(".hol"):
            colored_parts.append(f"[bright_green]{part}[/]")
        elif part in ["dry", "train", "dump", "new", "resume", "vllm"]:
            # Commands - bright cyan
            colored_parts.append(f"[bright_cyan]{part}[/]")
        elif part.startswith("--"):
            colored_parts.append(f"[yellow]{part}[/]")
        elif part.startswith("-"):
            colored_parts.append(f"[bright_yellow]{part}[/]")
        elif part.isdigit():
            colored_parts.append(f"[magenta]{part}[/]")
        elif ":" in part and ("." in part or "localhost" in part or part.startswith("127.")):
            # IP addresses, URLs, or localhost:port
            colored_parts.append(f"[magenta]{part}[/]")
            # elif part == "<loom/holoware/session_name>":
            # Special handling for the complex placeholder
            colored_text = "<" + loom("loom").replace("[/]", "") + "/" + \
                           holoware("holoware").replace("[/]", "") + "/" + \
                           session_name("session_name").replace("[/]", "") + ">"
            colored_parts.append(colored_text)
        elif part == "<command>":
            # Special handling for command placeholder
            colored_text = "<" + command_name("command").replace("[/]", "") + ">"
            colored_parts.append(colored_text)
        elif part == "<session_name>":
            # Special handling for session_name placeholder
            colored_text = "<" + session_name("session_name").replace("[/]", "") + ">"
            colored_parts.append(colored_text)
        elif part.startswith("<") and part.endswith(">"):
            # Generic placeholders
            colored_parts.append(placeholder(part))
        elif part == "..." or part == "…":
            # Ellipsis for abbreviated commands
            colored_parts.append(f"[dim]{part}[/]")
        else:
            colored_parts.append(f"[white]{part}[/]")

    return " ".join(colored_parts)


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
