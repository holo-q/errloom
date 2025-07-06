import inspect
import logging
import sys
import textwrap
from typing import Dict, Optional, Tuple

from verifiers.holoware.holophore import Holophore
from verifiers.holoware.holoware import ClassSpan, ContextResetSpan, HoloSpan, Holoware, ObjSpan, EgoSpan, SamplerSpan, TextSpan

logger = logging.getLogger(__name__)

# --- Holoware Grammar Definition ---
# This table defines the parsing rules for the Holoware DSL.
# The parser iterates through these rules and uses the first one that matches a given tag.

def _create_class_span(base_tag, node_attrs, positional_args) -> list[HoloSpan]:
    """Handler for creating ClassSpans."""
    class_span = ClassSpan(class_name=base_tag)
    class_span.apply_arguments(node_attrs, {}, positional_args)
    return [class_span]

def _create_role_or_sampler_span(base_tag, node_attrs, positional_args) -> list[HoloSpan]:
    """Handler for role tags, which may also create an associated SamplerSpan."""
    role = 'assistant' if base_tag.startswith('@_@') else 'user'
    has_sampler_attrs = any(attr in node_attrs for attr in ['goal', 'fence']) or positional_args

    node_id = ""
    if ':' in base_tag:
        _, node_id = base_tag.split(':', 1)

    spans = []
    if has_sampler_attrs:
        spans.append(EgoSpan(ego=role))
        sampler_span = SamplerSpan(id=node_id.strip())
        sampler_span.apply_arguments(node_attrs, {}, positional_args)
        spans.append(sampler_span)
    else:
        role_span = EgoSpan(id=node_id.strip(), ego=role)
        role_span.apply_arguments(node_attrs, {}, positional_args)
        spans.append(role_span)
    return spans

def _create_context_reset_span(train: bool):
    """Factory for creating context reset handlers."""

    def handler(base_tag, node_attrs, positional_args) -> list[HoloSpan]:
        return [ContextResetSpan(train=train)]

    return handler

def _create_obj_span(tag, node_attrs, positional_args) -> list[HoloSpan]:
    """Fallback handler for creating ObjSpans from one or more IDs."""
    var_ids = [v.strip() for v in tag.split('|')]
    obj_span = ObjSpan(var_ids=var_ids)
    obj_span.apply_arguments(node_attrs, {}, positional_args)
    return [obj_span]


HOLOWARE_GRAMMAR = [
    {
        'name':        "Class Span",
        'description': "A tag starting with an uppercase letter, e.g., <|MyClass arg|>. Must be a valid Python identifier.",
        'match':       lambda bt, na, pa: bt and bt[0].isupper() and bt.isidentifier(),
        'handler':     _create_class_span
    },
    {
        'name':        "Role/Sampler Span",
        'description': "A role tag for user <|o_o|> or assistant <|@_@|>. Can include a colon for an ID and attributes to become a sampler, e.g., <|@_@:id goal=...|>",
        'match':       lambda bt, na, pa: bt.startswith('@_@') or bt.startswith('o_o'),
        'handler':     _create_role_or_sampler_span
    },
    {
        'name':        "Training Context Reset",
        'description': "The <|+++|> tag, which clears context and marks the next conversation for training.",
        'match':       lambda bt, na, pa: bt == '+++',
        'handler':     _create_context_reset_span(train=True)
    },
    {
        'name':        "Standard Context Reset",
        'description': "The <|===|> tag, which clears context without marking it for training.",
        'match':       lambda bt, na, pa: bt == '===',
        'handler':     _create_context_reset_span(train=False)
    }
]

# --- End of Grammar Definition ---

class HolowareParser:
    def __init__(self, content: str):
        self.content = content
        self.pos = 0
        self.holoware = Holoware()

    def parse(self) -> "Holoware":
        """
        Parse a holoware DSL string into a Holoware object.
        """
        filtered_content = filter_comments(self.content)
        return self._parse_prompt(filtered_content)


    @classmethod
    def _read_and_parse_indented_block(cls, content: str, start_pos: int) -> tuple[Optional["Holoware"], int]:
        logger.debug(f"--- Checking for indented block at pos {start_pos} ---")

        lines_after = content[start_pos:].splitlines(True)

        if not lines_after:
            logger.debug("No lines after cursor, not a block.")
            return None, start_pos

        first_line_with_content_idx = -1
        for i, line in enumerate(lines_after):
            if line.strip():
                first_line_with_content_idx = i
                break

        if first_line_with_content_idx == -1:
            logger.debug("No content lines after cursor, not a block.")
            return None, start_pos

        # Determine the indentation from the first real line
        indentation = lines_after[first_line_with_content_idx]
        indent_level = len(indentation) - len(indentation.lstrip(' '))
        logger.debug(f"First content line has indent level {indent_level}.")
        if indent_level == 0:
            logger.debug("Not an indented block.")
            return None, start_pos

        # Collect all lines that belong to the indented block
        block_lines = []
        consumed_chars = 0

        # Consume any blank lines before the first content
        for i in range(first_line_with_content_idx):
            consumed_chars += len(lines_after[i])

        # Now, collect the indented lines
        logger.debug("Starting to consume indented lines.")
        for line in lines_after[first_line_with_content_idx:]:
            current_indent = len(line) - len(line.lstrip(' ')) if line.strip() else indent_level
            if line.strip() and current_indent < indent_level:
                logger.debug(f"Dedented line found, stopping block collection: '{line.strip()}'")
                break

            block_lines.append(line)
            consumed_chars += len(line)
        logger.debug(f"Consumed {len(block_lines)} lines for the block.")

        # Reconstruct and dedent the block
        dedented_text = textwrap.dedent("".join(block_lines))
        if not dedented_text.strip():
            logger.debug("Block was only whitespace, ignoring.")
            return None, start_pos

        logger.debug(f"Parsing nested block of {len(dedented_text)} chars. End pos: {start_pos + consumed_chars}")
        # Parse the dedented block recursively
        inner_holoware = HolowareParser(dedented_text).parse()

        return inner_holoware, start_pos + consumed_chars

    def _parse_prompt(self, content: str) -> "Holoware":
        """
        Parse a string containing the prompt DSL into a PromptTemplate object.
        Uses manual string parsing instead of regex for better maintainability.
        """
        ret = Holoware()
        pos = 0
        content_len = len(content)

        def read_until_tag_end() -> str:
            nonlocal pos
            start = pos
            while pos < content_len:
                if content[pos:pos + 2] == '|>':
                    pos += 2
                    return content[start:pos - 2]
                pos += 1
            raise ValueError(f"Unclosed tag at position {start}")

        def parse_attributes(tag: str) -> Tuple[Dict[str, str], list[str], str]:
            node_attrs: Dict[str, str] = {}
            positional_args: list[str] = []
            parts = tag.split()
            base_tag = parts[0]

            for part in parts[1:]:
                if "=" in part:
                    name, val = part.split('=', 1)
                    node_attrs[name.lower()] = val
                elif part.startswith("<>"):
                    if len(part) > 2:
                        node_attrs['fence'] = part[2:]
                    else:
                        raise ValueError("Empty <> attribute")
                else:
                    positional_args.append(part)

            return node_attrs, positional_args, base_tag

        def parse_tag(tag: str, out: list) -> None:
            """Finds the correct handler in the grammar and creates span(s) for a tag."""
            tag_content = tag.strip()
            node_attrs, positional_args, base_tag = parse_attributes(tag_content)

            for rule in HOLOWARE_GRAMMAR:
                if rule['match'](base_tag, node_attrs, positional_args):
                    spans = rule['handler'](base_tag, node_attrs, positional_args)
                    out.extend(spans)
                    return

            # Fallback to ObjSpan if no other rule matches
            spans = _create_obj_span(tag, node_attrs, positional_args)
            out.extend(spans)

        def read_text_content() -> str:
            nonlocal pos
            start = pos
            while pos < content_len:
                if content[pos:pos + 2] == '<|':
                    break
                pos += 1
            return content[start:pos]

        role = None
        out = []

        while pos < content_len:
            if content[pos:pos + 2] == '<|':
                pos += 2
                tag_content = read_until_tag_end()
                parse_tag(tag_content, out)

                last_span_in_ret = (ret.spans or [None])[-1]
                if isinstance(last_span_in_ret, ClassSpan):
                    body_holoware, new_pos = self._read_and_parse_indented_block(content, pos)
                    if body_holoware:
                        last_span_in_ret.body = body_holoware
                        pos = new_pos
                        continue
            else:
                text = read_text_content()
                if text:
                    out.append(TextSpan(text=text))

            spans_to_commit = []
            for span in out:
                is_text = isinstance(span, TextSpan)
                last_span = (ret.spans or spans_to_commit or [None])[-1]

                if not role and is_text and span.text.strip():
                    new_role = EgoSpan(ego='system')
                    spans_to_commit.append(new_role)
                    role = 'system'
                    last_span = new_role

                if isinstance(span, EgoSpan):
                    if role == span.ego:
                        continue
                    role = span.ego
                elif isinstance(span, ContextResetSpan):
                    role = None
                elif not role and isinstance(span, (SamplerSpan, ObjSpan, ClassSpan)):
                    raise ValueError(f"Cannot have {type(span).__name__} before a role.")
                elif not role and is_text and span.text.strip():
                    raise ValueError(f"Cannot have text span before a role: '{span.display_text}'")

                if is_text and isinstance(last_span, EgoSpan):
                    span.text = span.text.lstrip()

                if is_text and not span.text.strip():
                    continue

                spans_to_commit.append(span)

            ret.spans.extend(spans_to_commit)
            out.clear()

        return ret

_holo_classes_cache = {}
_cache_valid = False

def find_holo_classes():
    global _holo_classes_cache, _cache_valid

    if _cache_valid:
        return _holo_classes_cache

    holo_classes = {}
    modules = dict(sys.modules)
    for module_name, module in modules.items():
        if module is None:
            continue

        try:
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if hasattr(obj, '__holo__') and callable(getattr(obj, '__holo__')):
                    holo_classes[name] = obj
        except Exception:
            continue

    _holo_classes_cache = holo_classes
    _cache_valid = True
    return holo_classes

def resolve_holo_class(class_name):
    holo_classes = find_holo_classes()
    return holo_classes.get(class_name)

def invalidate_holo_cache():
    global _cache_valid
    _cache_valid = False

def enhanced_obj_span_processing(node: ObjSpan, env: dict, holophore: Holophore) -> Optional[str]:
    val = None
    for var_id in node.var_ids:
        val = env.get(var_id)
        if val is not None:
            break

    if val is None:
        return f"\n<obj id='{node.var_ids[0]}'>{str(val or 'NOT_FOUND')}</obj>"

    return str(val)

def enhanced_class_span_processing(node: ClassSpan, env: dict, holophore: Holophore) -> Optional[str]:
    val = env.get(node.class_name)
    if val is None:
        val = resolve_holo_class(node.class_name)
        if val:
            logger.debug(f"Found class {node.class_name} across loaded modules")
            env[node.class_name] = val

    if inspect.isclass(val) and hasattr(val, '__holo__'):
        _holo_ = val.__holo__
        sig = inspect.signature(_holo_)

        if len(sig.parameters) >= 2:
            try:
                return _holo_(holophore, node, *node.var_args, **node.var_kwargs)
            except TypeError as e:
                logger.debug(f"Failed to call {node.class_name}.__holo__ with extra arguments: {e}")
                try:
                    return _holo_(holophore, node)
                except TypeError:
                    logger.debug(f"Failed to call {node.class_name}.__holo__ with holophore/span, falling back to no args")
                    return _holo_()
        else:
            return _holo_()
    else:
        return f"\n<class name='{node.class_name}'>NOT_FOUND</class>"

def filter_comments(content: str) -> str:
    """
    Remove comment lines (starting with #) from prompt content.
    """
    lines = content.split('\n')
    # Rebuild the content string, ignoring any line that starts with '#'
    filtered_lines = [line for line in lines if not line.strip().startswith('#')]
    return '\n'.join(filtered_lines)
