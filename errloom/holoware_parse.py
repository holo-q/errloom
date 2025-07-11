import logging
import shlex
import textwrap
from typing import Dict, Optional, Tuple, TypeAlias

from rich.panel import Panel
from rich.text import Text as PrintedText

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

logger = logging.getLogger(__name__)

# --- Grammar Definition ---

def is_ego_or_sampler(base, kargs, kwargs) -> bool:
    return base in ("o_o", "@_@", "x_x") or "goal" in kwargs

def is_context_reset(base, kargs, kwargs) -> bool:
    return base in ("+++", "===")

HOLOWARE_GRAMMAR: list[Dict] = [
    {"match": is_ego_or_sampler, "handler": lambda *args: _build_ego_or_sampler_span(*args)},
    {"match": is_context_reset, "handler": lambda *args: _build_context_reset_span(train=args[1] == "+++")(*args)},
]

# --- Span Builders ---

def _build_class_span(out: list[Span], base, kargs, kwargs):
    """Handler for creating ClassSpans."""
    span = ClassSpan(class_name=base)
    span.set_args(kargs, kwargs, {})
    span.body = None
    out.append(span)

def _build_ego_or_sampler_span(out: list[Span], base, kargs, kwargs):
    """Handler for creating EgoSpans and optionally SampleSpans."""
    has_ego = any(e in base for e in ("o_o", "@_@", "x_x"))
    
    if has_ego:
        parts = base.split(':', 1)
        ego_str = parts[0]
        ego_map = {"o_o": "user", "@_@": "assistant", "x_x": "system"}
        
        ego_span = EgoSpan(ego=ego_map[ego_str])
        if len(parts) > 1:
            ego_span.uuid = parts[1]

        out.append(ego_span)

    if "goal" in kwargs:
        uuid = base if not has_ego else (base.split(':', 1)[1] if ':' in base else "")
        span = SampleSpan(uuid=uuid).set_args(kargs, kwargs, {})
        out.append(span)

def _build_context_reset_span(train: bool):
    def _builder(out: list[Span], base, kargs, kwargs):
        """Handler for creating ContextResetSpans."""
        out.append(ContextResetSpan(train=train))

    return _builder

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

    logger.debug(spantext)

    for rule in HOLOWARE_GRAMMAR:
        if rule['match'](base, kargs, kwargs):
            rule['handler'](out, base, kargs, kwargs)
            return

    # Fallback for ObjSpans or unhandled ClassSpans
    if base:
        if base[0].isupper():
            _build_class_span(out, base, kargs, kwargs)
        else:
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
    def __init__(self, code: str):
        self.code = code
        self.pos = 0
        self.ware = Holoware()
        self.ego: Optional[str] = None

    def parse(self) -> "Holoware":
        logger.info("parsing holoware ...")

        self.code = filter_comments(self.code)
        
        if self.code.strip() and not self.code.lstrip().startswith('<|'):
            self.ego = 'system'
            self.ware.spans.append(EgoSpan(ego='system'))

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

    def _parse_span(self, spantext: str):
        spanbuf = []
        build_span(spanbuf, spantext)
        for span in spanbuf:
            self._add_span(span)
            
        # check for an indented block to be its body.
        last_span = self.ware.spans[-1] if self.ware.spans else None
        if isinstance(last_span, ClassSpan):
            body_holoware, new_pos = self._read_and_parse_indented_block(self.code, self.pos)
            if body_holoware:
                last_span.body = body_holoware
                self.pos = new_pos

    def _parse_text(self, text: str):
        if not text:
            return

        # Unescape backslashes and tags
        processed_text = text.replace('\\\\', '\\').replace('\\<|', '<|')

        if not processed_text:
            return

        span = TextSpan(text=processed_text)
        self._add_span(span)

    def _find_next_span_start(self) -> int:
        pos = self.pos
        while True:
            pos = self.code.find("<|", pos)
            if pos == -1:
                return -1

            num_backslashes = 0
            i = pos - 1
            while i >= 0 and self.code[i] == '\\':
                num_backslashes += 1
                i -= 1
            
            if num_backslashes % 2 == 1:
                # Escaped, continue searching
                pos += 1
                continue
            
            return pos

    def _read_and_parse_indented_block(self, code: str, start_pos: int) -> Tuple[Optional[Holoware], int]:
        if start_pos >= len(code) or code[start_pos] != '\n':
            return None, start_pos

        line_start = start_pos + 1
        if line_start >= len(code):
            return None, start_pos

        first_line_end = code.find('\n', line_start)
        if first_line_end == -1:
            first_line_end = len(code)

        first_line = code[line_start:first_line_end]
        indent_level = len(first_line) - len(first_line.lstrip(' '))
        if indent_level == 0:
            return None, start_pos

        block_lines = [first_line.lstrip()]
        consumed_chars = len(first_line) + 1

        current_pos = first_line_end + 1
        while current_pos < len(code):
            next_line_end = code.find('\n', current_pos)
            if next_line_end == -1:
                next_line_end = len(code)

            line = code[current_pos:next_line_end]
            if not line.strip():
                block_lines.append("")
                consumed_chars += len(line) + 1
                current_pos = next_line_end + 1
                continue

            current_indent = len(line) - len(line.lstrip(' '))
            if current_indent < indent_level:
                break

            block_lines.append(line[indent_level:])
            consumed_chars += len(line) + 1
            current_pos = next_line_end + 1

        dedented_text = "\n".join(block_lines)

        panel_text = (
            f"Parsing Indented Block\n"
            f"Start Position: {start_pos}\n"
            f"Indent Level: {indent_level}\n"
            f"Consumed Chars: {consumed_chars}\n"
            f"Block Content:\n{dedented_text}"
        )
        logger.debug(Panel(panel_text))

        # Parse the dedented block recursively
        innerware = HolowareParser(dedented_text.rstrip()).parse()

        return innerware, start_pos + consumed_chars

    def read_until_span_end(self) -> str:
        if self.code[self.pos:self.pos+2] != '<|':
            raise ValueError("Not at the start of a span")

        start = self.pos + 2
        try:
            end = self.code.index("|>", start)
            self.pos = end + 2
            return self.code[start:end]
        except ValueError:
            raise ValueError("Unclosed tag")


