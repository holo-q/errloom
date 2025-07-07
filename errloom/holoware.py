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
import inspect
import logging
import typing
import uuid
from dataclasses import dataclass, field, fields
from typing import Optional

from rich.text import Text

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from errloom.holophore import Holophore

def holostatic(cls):
    """
    A class decorator that marks a class as holostatic.
    Holostatic classes have their __holo__ methods treated as class methods.
    They are not instantiated, and the class itself is passed to the holofunction.
    """
    setattr(cls, '_is_holostatic', True)
    return cls

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
    args: list[str] = field(default_factory=list)
    kwargs: dict[str, str] = field(default_factory=dict)

    def apply_arguments(self, node_attrs: dict[str, str], fence_attrs: dict[str, str], positional_args: list[str]) -> None:
        """Apply parsed arguments to span attributes.

        This method sets span attributes from parsed arguments, with the following priority:
        1. Set known span fields from node_attrs if they exist as fields
        2. Store remaining node_attrs and fence_attrs in var_kwargs
        3. Store positional_args in var_args
        """
        # Store positional args
        self.args = positional_args.copy()

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
        self.kwargs = {**remaining_attrs, **fence_attrs}

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
        if span.args:
            args_text.append(", ".join(map(str, span.args)))
        if span.kwargs:
            args_text.append(", ".join(f"{k}={v}" for k, v in span.kwargs.items()))

        if args_text:
            text.append(" | ", style="dim white")
            text.append(", ".join(args_text), style="magenta")

    def to_rich(self, indent_level=0) -> Text:
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
                    text.append(span.body.to_rich(indent_level + 1))
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

    def __call__(self, holophore: "Holophore"):
        from errloom.rollout import Context

        # it is require since up above its only imported for type_checking whith ""
        # it is literally the case that both pyright and pycharm's own inspections
        # simply miss this and do not understand it. Behold the failures of modern
        # software engineering:
        # noinspection PyUnresolvedReferences
        from errloom.holophore import Holophore  # noqa: F401

        # TODO we should move the span handler code to SpanHandlers - we want to keep the code separate from the data structure to keep the emphasis and clarity from both angles
        # holophore = Holophore(loom, roll, env)
        holophore.contexts.append(Context())

        _span_by_uuid:dict[str, HoloSpan] = {span.uuid: span for span in self.spans}

        def _call_holofunc(target, funcname, args, kwargs, optional=True, filter_missing_arguments=True):
            """
            Walks the MRO of a class or instance to find and call a __holo__ method
            from its defining base class.
            If `filter_missing_arguments` is True, it inspects the function signature
            and only passes keyword arguments that are expected by the function.
            """
            if funcname == '__init__':
                if not isinstance(target, type):
                    raise TypeError(f"Target for __init__ must be a class, not {type(target)}")

                final_kwargs = kwargs
                if filter_missing_arguments:
                    sig = inspect.signature(target)
                    final_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
                return target(*args, **final_kwargs)

            Impl = target if isinstance(target, type) else type(target)
            for Base in Impl.__mro__:
                if funcname in Base.__dict__:
                    logger.debug("%s.%s", Base, funcname)
                    # if args:
                    #     logger.debug(PrintedText(args))
                    # if kwargs:
                    #     logger.debug(PrintedText(kwargs))
                    _holofunc_ = getattr(Base, funcname)

                    final_kwargs = kwargs
                    if filter_missing_arguments:
                        sig = inspect.signature(_holofunc_)
                        final_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

                    return _holofunc_(target, *args, **final_kwargs)

            if not optional:
                raise AttributeError(f"No {funcname} method found in MRO for {Impl}")

            return None

        def _get_holofunc_args(span: HoloSpan):
            return [holophore, span], {}

        # --- Holo Lifecycle: Start ---
        # ----------------------------------------
        errors = 0

        for span in self.spans:
            if isinstance(span, ClassSpan):
                ClassName = span.class_name
                Class = holophore.env.get(ClassName)
                kspan, kwspan = _get_holofunc_args(span)
                if not Class:
                    from errloom.discovery import get_class
                    Class = get_class(ClassName)

                if not Class:
                    logger.error(f"Class '{ClassName}' not found in environment or registry.")
                    errors += 1
                    continue

                if getattr(Class, '_is_holostatic', False):
                    holophore.context.holofunc_targets[span.uuid] = Class
                else:
                    inst = _call_holofunc(Class, '__init__', kspan, kwspan, optional=False)
                    holophore.context.holofunc_targets[span.uuid] = inst

                    # try:
                    inst_after_init = _call_holofunc(inst, '__holo_init__', kspan, kwspan)
                    if inst_after_init:
                        holophore.context.holofunc_targets[span.uuid] = inst_after_init

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
            if isinstance(span, (TextSpan, ObjSpan)):
                if isinstance(span, TextSpan):
                    buf.append(span.text)
                elif isinstance(span, ObjSpan):
                    for var_id in span.var_ids:
                        if var_id in holophore.env:
                            value = holophore.env[var_id]
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
                sample = holophore.sample(holophore.rollout)
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
                ClassName = span.class_name
                Class = holophore.env.get(ClassName)
                if not Class:
                    from errloom.discovery import get_class
                    Class = get_class(ClassName)

                if not Class:
                    # This should have been caught in the init loop, but as a safeguard:
                    logger.error(f"Class '{ClassName}' not found for __holo__ call.")
                    continue

                kspan, kwspan = _get_holofunc_args(span)

                # TODO this should be handled by the individual HoloSpans. They use the body however they want, can make it conditional, etc.
                # if span.body:
                #     phoron = span.body(roll, sample_callback, env=env)
                #     holo_kwargs['body'] = phoron

                try:
                    result = None
                    inst = holophore.context.holofunc_targets.get(span.uuid, None)
                    if inst:
                        result = _call_holofunc(inst, '__holo__', kspan, kwspan, optional=False)

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
        for uid, target in holophore.context.holofunc_targets.items():
            span = _span_by_uuid[uid]
            kspan, kwspan = _get_holofunc_args(span)
            if hasattr(target, '__holo_end__'):
                try:
                    _call_holofunc(target, '__holo_end__', kspan, kwspan)
                except Exception as e:
                    target_name = getattr(target, '__name__', target.__class__.__name__)
                    logger.error(f"Failed to execute __holo_end__ for {target_name}: {e}", exc_info=True)

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

    def first_span_by_type(self, SpanType) -> HoloSpan:  # TODO can we annotate that the return type is of SpanType? it's basically a generic we want....
        for span in self.spans:
            if isinstance(span, SpanType):
                return span
        raise ValueError(f"Could not find span matching type {SpanType}")  # TODO better exception type

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
