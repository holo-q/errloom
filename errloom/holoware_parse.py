import shlex
import textwrap
from typing import Dict, Optional, Tuple

from rich.panel import Panel

from errloom.lib.log import indent_decorator
from .lib import log
from .holoware import (
    ClassSpan,
    ContextResetSpan,
    EgoSpan,
    Holoware,
    ObjSpan,
    SampleSpan,
    Span,
    TextSpan,
)

logger = log.getLogger(__name__)

# --- Grammar Definition ---
def is_ego_or_sampler(base, kargs, kwargs) -> bool:
    return base in ("o_o", "@_@", "x_x") or "goal" in kwargs or "<>" in kwargs

def is_context_reset(base, kargs, kwargs) -> bool:
    return base in ("+++", "===", "---", "^^^", "###", "@@@", "\"\"\"", "***", "%%%")

HOLOWARE_GRAMMAR: list[Dict] = [
    {"match": is_ego_or_sampler, "handler": lambda *args: _build_ego_or_sampler_span(*args)},
    {"match": is_context_reset, "handler": lambda *args: _build_context_reset_span(train=args[1] == "+++")(*args)},
]

EGO_MAP = {"o_o": "user", "@_@": "assistant", "x_x": "system"}

# --- Span Builders ---

def _build_class_span(out: list[Span], base, kargs, kwargs):
    """Handler for creating ClassSpans."""
    span = ClassSpan(class_name=base)
    span.set_args(kargs, kwargs, {})
    span.body = None
    out.append(span)

def _build_ego_or_sampler_span(out: list, base: str, kargs: list, kwargs: dict):
    # Split ego and potential UUID
    parts = base.split(":", 1)
    ego = parts[0]
    uuid = parts[1] if len(parts) > 1 else ""

    out.append(EgoSpan(ego=EGO_MAP.get(ego, ego), uuid=uuid))

    # A sampler can be defined with kargs/kwargs on the ego span
    sampler_kwargs = {k: v for k, v in kwargs.items() if k not in ("<>", "goal")}
    goal = kwargs.get("<>", "") or kwargs.get("goal", "")
    if kargs or sampler_kwargs or goal:
        if not goal:
            # It's not a sampler if it doesn't have a goal
            # This can happen if there are only kargs, which are not supported for samplers.
            # We just ignore them.
            pass
        else:
            out.append(SampleSpan(uuid=uuid, kargs=kargs, kwargs=sampler_kwargs, goal=goal))


def _build_context_reset_span(train: bool):
    def _handler(out: list, base: str, kargs: list, kwargs: dict):
        """Handler for creating ContextResetSpans."""
        out.append(ContextResetSpan(train=train))

    return _handler

def _build_obj_span(out: list[Span], base, kargs, kwargs):
    """Fallback handler for creating ObjSpans from one or more IDs."""
    logger.debug("")
    var_ids = [v.strip() for v in base.split('|')]
    span = ObjSpan(var_ids=var_ids).set_args(kargs, kwargs, {})
    out.append(span)

def build_span(out: list[Span], spantext: str):
    """Finds the correct handler in the grammar and creates span(s) for a tag."""
    tag_content = spantext.strip()
    if not tag_content:
        return
    base, kargs, kwargs = parse_span_tag(tag_content)

    logger.debug(f"base='{base}' kargs={kargs} kwargs={kwargs}")

    for rule in HOLOWARE_GRAMMAR:
        if rule['match'](base, kargs, kwargs):
            logger.debug(f"handler={rule['match'].__name__}")
            rule['handler'](out, base, kargs, kwargs)
            return

    # Fallback for ObjSpans or unhandled ClassSpans
    if base:
        if base[0].isupper():
            logger.debug("handler=_build_class_span ClassSpan")
            _build_class_span(out, base, kargs, kwargs)
        else:
            logger.debug("handler=_build_class_span ObjSpan")
            _build_obj_span(out, base, kargs, kwargs)

def parse_span_tag(tag: str) -> Tuple[str, list[str], Dict[str, str]]:
    kwargs: Dict[str, str] = {}
    kargs: list[str] = []

    parts = shlex.split(tag)
    if not parts:
        return "", [], {}

    base_span = parts[0]
    new_kargs = []
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            kwargs[key] = value
        elif part.startswith('<>'):
            if len(part) > 2:
                kwargs['<>'] = part[2:]
            else:
                raise ValueError("Empty <> attribute")
        else:
            new_kargs.append(part)

    kargs.extend(new_kargs)

    return base_span, kargs, kwargs


def filter_comments(content: str) -> str:
    """Removes comments from holoware content."""
    lines = content.split('\n')
    processed_lines = []
    for line in lines:
        if line.strip().startswith('#'):
            processed_lines.append("")
        else:
            processed_lines.append(line)
    return "\n".join(processed_lines)


class HolowareParser:
    def __init__(self, code: str, ego=None):
        self.code = code
        self.pos = 0
        self.ware = Holoware()
        self.ego: Optional[str] = ego

    def parse(self) -> "Holoware":
        logger.push_debug("PARSE")
        self.code = filter_comments(self.code)

        # if self.code.strip() and not self.code.lstrip().startswith('<|'):
        #     self.ego = 'system'
        #     self.ware.spans.append(EgoSpan(ego='system'))

        while self.pos < len(self.code):
            next_span_pos = self._find_next_span_start()

            if next_span_pos == -1:
                self._parse_text(self.code[self.pos:])
                break

            if next_span_pos > self.pos:
                self._parse_text(self.code[self.pos:next_span_pos])

            self.pos = next_span_pos
            spantext = self.read_until_span_end()
            self._parse_span(spantext)

        self._add_implicit_ego_if_needed()
        self._finalize_text_span()

        logger.pop()
        return self.ware

    def _add_implicit_ego_if_needed(self):
        # Implicitly create a system ego if there's text content before any ego is set.
        if self.ego:
            return

        text_span_found = False
        for s in self.ware.spans:
            if isinstance(s, TextSpan):
                if s.text.strip():
                    text_span_found = True
                break
            if isinstance(s, (EgoSpan, ContextResetSpan)):
                # Don't add implicit ego if an explicit one is already there
                return

        if text_span_found:
            new_ego = EgoSpan(ego='system')
            self.ware.spans.insert(0, new_ego)
            self.ego = 'system'

    def _add_span(self, span: Span):
        """Adds a span to the current holoware object."""
        logger.debug(f"[bold green]{type(span).__name__}[/]")
        last_span = self.ware.spans[-1] if self.ware.spans else None
        is_text = isinstance(span, TextSpan)

        if not self.ego and is_text and span.text.strip():
            new_ego = EgoSpan(ego='system')
            self.ware.spans.append(new_ego)
            self.ego = new_ego.ego
            last_span = new_ego

        if not self.ego and isinstance(span, (SampleSpan, ObjSpan, ClassSpan)):
            self._add_implicit_ego_if_needed()
            if not self.ego:
                raise ValueError(f"Cannot have {type(span).__name__} before a ego.")

        if isinstance(span, EgoSpan):
            # Avoid creating duplicate, consecutive ego spans.
            if self.ego == span.ego:
                return
            self.ego = span.ego
        elif isinstance(span, ContextResetSpan):
            self.ego = None  # Context resets clear the current ego.

        # Merge consecutive text spans for cleaner output.
        if is_text and isinstance(last_span, TextSpan):
            last_span.text += span.text
            return

        # Remove leading whitespace from text that follows any span to keep it clean.
        if is_text and not isinstance(last_span, TextSpan) and last_span is not None:
            span.text = span.text.lstrip()

        # Don't add text spans that are only whitespace
        if is_text and not span.text.strip():
            return

        self.ware.spans.append(span)

    @indent_decorator("SPAN", log_func=logger.debug)
    def _parse_span(self, spantext: str):
        # log.push("PARSE", f"<|{spantext}|>")
        spanbuf = []
        build_span(spanbuf, spantext)
        for span in spanbuf:
            self._add_span(span)

        # check for an indented block to be its body.
        last_span = self.ware.spans[-1] if self.ware.spans else None
        if isinstance(last_span, ClassSpan):
            body_holoware, new_pos = self._parse_indented_block(self.code, self.pos)
            if body_holoware:
                last_span.body = body_holoware
                self.pos = new_pos
        # log.pop()

    @indent_decorator("TEXT", log_func=logger.debug)
    def _parse_text(self, text: str):
        if not text:
            return

        # Unescape backslashes and tags
        processed_text = text.replace('\\\\', '\\').replace('\\<|', '<|')

        if not processed_text:
            return

        logger.debug(f"{repr(processed_text[:40])}")
        span = TextSpan(text=processed_text)
        self._add_span(span)

    def _find_next_span_start(self) -> int:
        pos = self.pos
        while True:
            found_pos = self.code.find("<|", pos)
            if found_pos == -1:
                logger.debug(f"no more spans from pos {self.pos}")
                return -1

            num_backslashes = 0
            i = found_pos - 1
            while i >= 0 and self.code[i] == '\\':
                num_backslashes += 1
                i -= 1

            if num_backslashes % 2 == 1:
                # Escaped, continue searching
                logger.debug(f"found escaped span at {found_pos}, continuing search from {found_pos + 1}")
                pos = found_pos + 1
                continue

            logger.debug(f"found span at {found_pos}")
            return found_pos

    @indent_decorator("BLOCK", log_func=logger.debug)
    def _parse_indented_block(self, code: str, start_pos: int) -> Tuple[Optional[Holoware], int]:
        block_content, end_pos = self._read_indented_block_content(code, start_pos)
        if block_content is None:
            logger.debug("[block] x (no content)")
            # log.pop()
            return None, start_pos

        # --- Dedent and Prepare for Parsing ---
        dedented_block = textwrap.dedent(block_content)

        logger.debug(Panel(
            dedented_block.strip(),
            title="Parsing Indented Block",
            border_style="dim",
            expand=False,
            padding=(1, 4)
        ))
        # Check if the block is empty or contains only whitespace
        if not dedented_block.strip():
            logger.debug("x (empty after dedent)")
            # log.pop()
            return None, end_pos

        # --- Recursive Parsing ---
        # Start a fresh parser for the block without inheriting the outer ego.
        # This allows the nested body to add an implicit system ego when appropriate.
        parser = HolowareParser(dedented_block)
        body = parser.parse()
        return body, end_pos

    def _read_indented_block_content(self, code: str, start_pos: int) -> Tuple[Optional[str], int]:
        """Reads an indented block of text, returning the content and new position."""
        lines = code[start_pos:].splitlines(True)
        if not lines:
            return None, start_pos

        first_line = lines[0]
        if not first_line.strip():  # Skip empty line after span
            start_pos += len(first_line)
            lines.pop(0)
            if not lines:
                return None, start_pos
            first_line = lines[0]

        indentation = len(first_line) - len(first_line.lstrip(' '))
        if indentation == 0:
            return None, start_pos

        block_lines = []
        current_pos = start_pos
        for line in lines:
            line_indent = len(line) - len(line.lstrip(' '))
            if line.strip() == "":  # allow empty lines within block
                block_lines.append(line)
                current_pos += len(line)
                continue
            if line_indent >= indentation:
                block_lines.append(line)
                current_pos += len(line)
            else:
                break

        if not block_lines:
            return None, start_pos

        return "".join(block_lines), current_pos

    def read_until_span_end(self) -> str:
        if self.code[self.pos:self.pos + 2] != '<|':
            raise ValueError("Not at the start of a span")

        start = self.pos + 2
        try:
            end = self.code.index("|>", start)
            self.pos = end + 2
            return self.code[start:end]
        except ValueError:
            raise ValueError("Unclosed tag")

    def _finalize_text_span(self):
        """Merges consecutive TextSpans and removes leading whitespace."""
        last_span = None
        for i, span in enumerate(self.ware.spans):
            if isinstance(span, TextSpan):
                if last_span and isinstance(last_span, TextSpan):
                    last_span.text += span.text
                    self.ware.spans.pop(i)  # Remove the duplicate
                else:
                    last_span = span
            else:
                last_span = span
