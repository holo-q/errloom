"""
Prompt Library Utilities

This module provides a robust parsing and formatting system for a custom prompt
DSL (Domain-Specific Language). It's designed to handle complex, multi-part
prompt structures that include not just conversational turns, but also metadata
for training and inference processes.

=== PROMPT DSL SYNTAX ===

The DSL uses a tag-based system with the `<|...|>` delimiter.

- Role Tags: Define the speaker in a conversation.
  - `<|o_o|>`: Sets role to user.
  - `<|@_@|>`: Sets role to assistant.
  - A system role is implicitly created for any text that directly follows a context reset tag.

- Data and Class Tags:
  - `<|var1|var2|>`: A placeholder for a data variable. The first variable found in the environment is used.
  - `<|CLASSNAME|>`: A `__holo__` injection point for a class. The class name must be uppercase.

- Attribute Syntax for samplers and classes:
  - `arg1 arg2`: Positional arguments.
  - `key=value`: Keyword arguments.
  - `<>name=val`: Special attributes for samplers (e.g., `<>fence=...`).

- Special Tags:
  - `<|===|>`: Clears the context.
  - `<|+++|>`: Clears the context and marks the next context for training.

The parser converts a prompt file into a structured `Holoware` object,
which contains a sequence of spans representing the parsed DSL.
"""
import logging
import uuid
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Optional

from rich.text import Text

from errloom.rollout import Rollout

logger = logging.getLogger(__name__)

# TODO type alias the openai conversation dict format
# TODO all union etc should use the type hinting syntax


# SPANS
# ----------------------------------------

@dataclass
class HoloSpan:
    """Base class for spans in the holoware DSL.

    A span represents a prompting segment with specific injection behavior.
    Spans can be role markers (user/assistant), context resets, variable placeholders,
    class injections, or sampling points. Each span type handles its section of the
    template differently during rendering.

    Spans can have:
    - An ID for referencing (defaults to UUID)
    - Variable arguments (positional)
    - Variable keyword arguments
    - Type-specific attributes defined in subclasses
    """
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    var_args: list[str] = field(default_factory=list)
    var_kwargs: dict[str, str] = field(default_factory=dict)

    def apply_arguments(self, node_attrs: dict[str, str], fence_attrs: dict[str, str], positional_args: list[str]) -> None:
        """Apply parsed arguments to span attributes.

        This method sets span attributes from parsed arguments, with the following priority:
        1. Set known span fields from node_attrs if they exist as fields
        2. Store remaining node_attrs and fence_attrs in var_kwargs
        3. Store positional_args in var_args
        """
        # Store positional args
        self.var_args = positional_args.copy()

        # Get all field names for this span class
        field_names = {f.name for f in fields(self)}

        # Apply node_attrs to matching fields, rest go to var_kwargs
        remaining_attrs = {}
        for key, value in node_attrs.items():
            if key in field_names:
                setattr(self, key, value)
            else:
                remaining_attrs[key] = value

        # Combine remaining node_attrs and fence_attrs
        self.var_kwargs = {**remaining_attrs, **fence_attrs}

@dataclass
class ContextResetSpan(HoloSpan):
    """Reset the context."""
    train: bool = False

@dataclass
class EgoSpan(HoloSpan):
    """Sets the current ego (role tag in OpenAI) by printing the special token."""
    ego: str = ""

@dataclass
class SamplerSpan(HoloSpan):
    """Sample tokens from the model"""
    goal: str = ""
    fence: str = ""

    # TODO we an add other arguments for temperature and stuff, which might not be static

    @property
    def display_goal(self):
        return self.goal[:30].replace('\n', '\\n').replace('\r', '\\r')

@dataclass
class TextSpan(HoloSpan):
    """Represents a block of plain text content."""
    text: str = ""

    @property
    def display_text(self):
        return self.text[:30].replace('\n', '\\n').replace('\r', '\\r')

@dataclass
class ObjSpan(HoloSpan):
    """Represents a data variable placeholder (e.g. <|sample|>)."""
    var_ids: list[str] = field(default_factory=list)

@dataclass
class ClassSpan(HoloSpan):
    """Represents a class with a __holo__ method."""
    class_name: str = ""
    body: Optional["Holoware"] = None

# HOLOWARE
# ----------------------------------------

@dataclass
class Holoware:
    """
    Structured representation of a parsed prompt template from the DSL.
    Contains a sequence of spans that define the entire prompt and its metadata.
    """
    spans: list[HoloSpan] = field(default_factory=list)

    @property
    def obj_ids(self) -> list[str]:
        """Returns a list of all object variable IDs referenced in the template."""
        ids = []
        for span in self.spans:
            if isinstance(span, ObjSpan):
                ids.extend(span.var_ids)
        return ids

    @property
    def trained_contexts(self) -> list[int]:
        """Returns a list of indices where context resets occur."""
        contexts = []
        current = 0
        for i, span in enumerate(self.spans):
            if isinstance(span, ContextResetSpan):
                if i > 0:  # if theres no context reset on the first line, then it's implicit
                    current += 1
                if span.train:
                    contexts.append(current)

        return contexts

    def _append_args_to_debug(self, text: Text, span: HoloSpan):
        """Append formatted var_args and var_kwargs to the rich text."""
        args_text = []
        if span.var_args:
            args_text.append(", ".join(map(str, span.var_args)))
        if span.var_kwargs:
            args_text.append(", ".join(f"{k}={v}" for k, v in span.var_kwargs.items()))

        if args_text:
            text.append(" | ", style="dim white")
            text.append(", ".join(args_text), style="magenta")

    def to_rich_debug(self, indent_level=0) -> Text:
        """Format the prompt template spans for rich debug display."""
        text = Text()
        if not self.spans:
            return text

        processed_indices = set()
        indent = "  " * indent_level

        # Calculate the width needed for span indices
        max_index = len(self.spans) - 1
        index_width = len(str(max_index))
        index_format = f"[{{:{index_width}d}}] "

        for i, span in enumerate(self.spans):
            if i in processed_indices:
                continue

            text.append(indent)
            text.append(index_format.format(i), style="dim white")

            if isinstance(span, ContextResetSpan):
                style = "bold yellow" if span.train else "yellow"
                text.append("ContextResetSpan", style=style)
                if span.train:
                    text.append(" (train=True)", style=style)
                self._append_args_to_debug(text, span)
                text.append("\n")

            elif isinstance(span, EgoSpan):
                text.append("RoleSpan (", style="bold cyan")
                role_display = span.ego.replace('\n', '\\n').replace('\r', '\\r')
                text.append(f"role={role_display}", style="cyan")
                self._append_args_to_debug(text, span)
                text.append(")", style="cyan")

                # Look ahead - if next span is a single TextSpan, combine them for compact view
                next_span_idx = i + 1
                if next_span_idx < len(self.spans) and isinstance(self.spans[next_span_idx], TextSpan):
                    next_span = self.spans[next_span_idx]
                    span_after_text_idx = next_span_idx + 1
                    span_after_text = self.spans[span_after_text_idx] if span_after_text_idx < len(self.spans) else None

                    # Pyright doesn't understand the type of next_span here, so we assert it
                    assert isinstance(next_span, TextSpan)

                    if not span_after_text or isinstance(span_after_text, (EgoSpan, ContextResetSpan)):
                        text.append(" → ", style="dim white")
                        text.append(f"'{next_span.display_text}{'...' if len(next_span.text) > 30 else ''}'", style="white")
                        self._append_args_to_debug(text, next_span)
                        processed_indices.add(next_span_idx)

                text.append("\n")

            elif isinstance(span, SamplerSpan):
                text.append("SamplerSpan (", style="bold green")
                parts = []
                if span.uuid:
                    parts.append(f"id={span.uuid}")
                if span.goal:
                    parts.append(f"goal={span.display_goal}...")
                if span.fence:
                    fence_display = span.fence.replace('\n', '\\n').replace('\r', '\\r')
                    parts.append(f"fence={fence_display}")
                text.append(", ".join(parts), style="green")
                self._append_args_to_debug(text, span)
                text.append(")\n", style="green")

            elif isinstance(span, TextSpan):
                text.append(f"TextSpan ('{span.display_text}{'...' if len(span.text) > 30 else ''}')", style="white")
                self._append_args_to_debug(text, span)
                text.append("\n")

            elif isinstance(span, ObjSpan):
                text.append("ObjSpan (", style="bold magenta")
                text.append(f"vars={span.var_ids}", style="magenta")
                self._append_args_to_debug(text, span)
                text.append(")\n", style="magenta")

            elif isinstance(span, ClassSpan):
                text.append("ClassSpan (", style="bold blue")
                text.append(f"class={span.class_name}", style="blue")
                self._append_args_to_debug(text, span)
                text.append(")\n", style="blue")
                if span.body:
                    text.append(span.body.to_rich_debug(indent_level + 1))
        return text

    @classmethod
    def parse(cls, content: str) -> "Holoware":
        """
        Parse a holoware DSL string into a Holoware object.

        Args:
            content: The holoware DSL string to parse

        Returns:
            A Holoware object representing the parsed template
        """
        from errloom.holoware_parse import HolowareParser
        return HolowareParser(content).parse()

    def __call__(self, roll: Rollout, sample_callback: Callable, env: Optional[dict[str, Any]] = None):
        from errloom.holophore import Holophore
        from errloom.rollout import Context

        if env is None:
            env = {}

        # TODO we should move the span handler code to SpanHandlers - we want to keep the code separate from the data structure to keep the emphasis and clarity from both angles

        holophore = Holophore(roll, env)
        holophore.contexts.append(Context())

        span_map = {span.uuid: span for span in self.spans}

        def _get_span_args(span):
            args = [holophore, span] + span.var_args
            kwargs = span.var_kwargs.copy()
            return args, kwargs

        # --- Holo Lifecycle: Start ---
        # ----------------------------------------
        errors = 0

        for span in self.spans:
            kspan, kwspan = _get_span_args(span)
            if isinstance(span, ClassSpan):
                ClassName = span.class_name
                Class = env.get(ClassName)
                if not Class:
                    from errloom.discovery import get_class
                    Class = get_class(ClassName)

                if not Class:
                    logger.error(f"Class '{ClassName}' not found in environment or registry.")
                    errors += 1
                    continue

                try:
                    if hasattr(Class, '__holo_init__'):
                        inst = Class.__holo_init__(*kspan, kwspan)
                        if inst:
                            holophore.context.class_instances[span.uuid] = inst

                except Exception as e:
                    logger.error(f"Failed to instantiate or run __holo_start__ for {ClassName}: {e}", exc_info=True)
                    errors += 1

        if errors > 0:
            raise RuntimeError(f"Failed to instantiate {errors} classes.")

        # STATE MANAGEMENT
        # ----------------------------------------
        ego = 'system'
        buf = []

        def flush():
            nonlocal buf
            if buf:
                # buftext <- buf
                buftext, buf = "".join(buf), []

                if holophore.context.messages and holophore.context.messages[-1]['role'] == ego:
                    holophore.context.messages[-1]['content'] += buftext
                else:
                    holophore.context.messages.append({'role': ego, 'content': buftext})

        # --- Holo Lifecycle: Main ---
        # ----------------------------------------

        for span in self.spans:
            kspan, kwspan = _get_span_args(span)

            if isinstance(span, (TextSpan, ObjSpan)):
                if isinstance(span, TextSpan):
                    buf.append(span.text)
                elif isinstance(span, ObjSpan):
                    for var_id in span.var_ids:
                        if var_id in env:
                            value = env[var_id]
                            buf.append(f"<obj id={var_id}>")
                            buf.append(str(value))
                            buf.append("</obj>")
                            break
                continue

            flush()

            if isinstance(span, ContextResetSpan):
                holophore.contexts.append(Context())
                ego = 'system'

            elif isinstance(span, EgoSpan):
                if ego != span.ego:
                    flush()
                    ego = span.ego

            elif isinstance(span, SamplerSpan):
                sample = sample_callback()
                if sample:
                    # If the SamplerSpan has a fence attribute, wrap the response in fence tags
                    if span.goal:
                        text = f"<{span.goal}>{sample}</{span.goal}>"
                    else:
                        text = sample

                    if holophore.context.messages and holophore.context.messages[-1]['role'] == 'assistant':
                        holophore.context.messages[-1]['content'] += '\n' + text
                    else:
                        holophore.context.messages.append({'role': 'assistant', 'content': text})

            elif isinstance(span, ClassSpan):
                inst = holophore.context.class_instances.get(span.uuid)
                if not inst:
                    logger.warning(f"No instance found for ClassSpan {span.class_name} ({span.uuid})")
                    continue

                if hasattr(inst, '__holo__'):
                    # Prepare arguments for __holo__

                    # TODO this should be handled by the individual HoloSpans. They use the body however they want, can make it conditional, etc.
                    # if span.body:
                    #     phoron = span.body(roll, sample_callback, env=env)
                    #     holo_kwargs['body'] = phoron

                    try:
                        result = inst.__holo__(*kspan, **kwspan)

                        # TODO what is this doing
                        if isinstance(result, list) and all(isinstance(m, dict) and 'role' in m for m in result):
                            holophore.context.messages.extend(result)
                        elif result is not None:
                            if 'holo_results' not in holophore.extra:
                                holophore.extra['holo_results'] = []
                            holophore.extra['holo_results'].append({span.class_name: result})
                    except Exception as e:
                        logger.error(f"Failed to execute __holo__ for {span.class_name}: {e}", exc_info=True)

        flush()

        # --- Holo Lifecycle: End ---
        # ----------------------------------------
        for span_id, inst in holophore.context.class_instances.items():
            span = span_map[span_id]
            kspan, kwspan = _get_span_args(span)
            if hasattr(inst, '__holo_end__'):
                try:
                    inst.__holo_end__(*kspan, **kwspan)
                except Exception as e:
                    logger.error(f"Failed to execute __holo_end__ for {inst.__class__.__name__}: {e}", exc_info=True)

        return holophore

    @classmethod
    def load(cls, filepath: str) -> "Holoware":
        """
        Load and parse a holoware file.

        Args:
            filepath: Path to the holoware file

        Returns:
            A Holoware object representing the parsed template
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return cls.parse(content)

# def format_prompt(input: Union[Holoware, str, MessageListStr], **kwargs) -> Union[str, MessageListStr]:
#     """Convenience function to format a prompt."""
#     # The template could be a legacy list, so we handle that case.
#     if isinstance(input, Holoware):
#         # The new formatter expects the data variables to be in the kwargs directly
#         messages = format_prompt(input, **kwargs)
#
#         if isinstance(messages, str):
#             return messages.format_map(kwargs)
#
#         # Now, perform the f-string style substitution
#         formatted_messages = []
#         for message in messages:
#             try:
#                 # The format_map is safer than format as it doesn't fail on extra keys
#                 formatted_content = message["content"].format_map(kwargs)
#                 formatted_messages.append({
#                     "role":    message["role"],
#                     "content": formatted_content
#                 })
#             except KeyError as e:
#                 logger.warning(f"Missing variable for formatting prompt: {e}")
#                 # Add the message with the placeholder to allow debugging
#                 formatted_messages.append(message)
#         return formatted_messages
#
#     # Handle legacy formats
#     if isinstance(input, list):
#         formatted_messages = []
#         for message in input:
#             formatted_content = message["content"].format(**kwargs)
#             formatted_messages.append({
#                 "role":    message["role"],
#                 "content": formatted_content
#             })
#         return formatted_messages
#     else:
#         return str(input).format(**kwargs)


# Cache for holo classes to avoid repeated module scanning
