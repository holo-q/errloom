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
import typing
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

from rich.text import Text

from errloom.utils import log

logger = log.getLogger(__name__)

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
class Span(ABC):
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
    kargs: list[str] = field(default_factory=list)
    kwargs: dict[str, str] = field(default_factory=dict)

    @abstractmethod
    def get_color(self) -> str:
        """Return the rich color style for this span type."""
        pass

    def set_args(self, k: list[str], kw: dict[str, str], kwinject: dict[str, str]) -> 'Span':
        """Apply parsed arguments to span attributes.

        This method sets span attributes from parsed arguments, with the following priority:
        1. Set known span fields from node_attrs if they exist as fields
        2. Store remaining node_attrs and fence_attrs in var_kwargs
        3. Store positional_args in var_args
        """
        # Store positional args
        self.kargs = k.copy()

        # Get all field names for this span class
        field_names = {f.name for f in fields(self)}

        # Apply node_attrs to matching fields; rest go to var_kwargs
        remaining = {}
        for key, value in kw.items():
            if key in field_names:
                setattr(self, key, value)
            else:
                remaining[key] = value

        # Combine remaining node_attrs and fence_attrs
        self.kwargs = {**remaining, **kwinject}
        return self

@dataclass
class ContextResetSpan(Span):
    """Reset the context."""
    train: bool = False

    def get_color(self) -> str:
        return "bold yellow" if self.train else "yellow"

@dataclass
class EgoSpan(Span):
    """Sets the current ego (role tag in OpenAI) by printing the special token."""
    ego: str = ""

    def get_color(self) -> str:
        return "bold cyan"

@dataclass
class SampleSpan(Span):
    """Sample tokens from the model"""
    goal: str = ""
    fence: str = ""

    # TODO we an add other arguments for temperature and stuff, which might not be static

    def get_color(self) -> str:
        return "bold green"

    @property
    def display_goal(self):
        return self.goal[:30].replace('\n', '\\n').replace('\r', '\\r')

@dataclass
class TextSpan(Span):
    """Represents a block of plain text content."""
    text: str = ""

    def get_color(self) -> str:
        return "white"

    @property
    def display_text(self):
        return self.text[:30].replace('\n', '\\n').replace('\r', '\\r')

@dataclass
class ObjSpan(Span):
    """Represents a data variable placeholder (e.g. <|sample|>)."""
    var_ids: list[str] = field(default_factory=list)

    def get_color(self) -> str:
        return "bold magenta"

@dataclass
class ClassSpan(Span):
    """Represents a class with a __holo__ method."""
    class_name: str = ""
    body: Optional["Holoware"] = None

    def get_color(self) -> str:
        return "bold blue"

# HOLOWARE
# ----------------------------------------

@dataclass
class Holoware:
    """
    Structured representation of a parsed prompt template from the DSL.
    Contains a sequence of spans that define the entire prompt and its metadata.
    """
    name: Optional[str] = None
    code: Optional[str] = None
    filepath: Optional[str] = None
    spans: list[Span] = field(default_factory=list)

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

    def first_span_by_type(self, SpanType) -> Span:  # TODO can we annotate that the return type is of SpanType? it's basically a generic we want....
        for span in self.spans:
            if isinstance(span, SpanType):
                return span
        raise ValueError(f"Could not find span matching type {SpanType}")  # TODO better exception type

    def __call__(self, phore: "Holophore"):
        from errloom.holoware_handlers import HolowareHandlers

        # it is require since up above its only imported for type_checking whith ""
        # it is literally the case that both pyright and pycharm's own inspections
        # simply miss this and do not understand it. Behold the failures of modern
        # software engineering:
        # noinspection PyUnresolvedReferences
        from errloom.holophore import Holophore  # noqa: F401

        logger.push_debug(f"WARE({self.name})" if self.name else "WARE")

        # --- Lifecycle: Start ---
        phore.invoke(self, "__holo_start__", [phore], {})
        phore.active_holowares.append(self)

        logger.push_debug("(start)")
        for span in self.spans:
            if isinstance(span, ClassSpan):
                classname = span.class_name
                Cls = phore.get_class(classname)

                if not Cls:
                    raise Exception(f"Class '{classname}' not found in environment or registry.")
                if getattr(Cls, '_is_holostatic', False):
                    phore.span_bindings[span.uuid] = Cls
                else:
                    # TODO extract to call_class_init (formalizes the api)
                    kspan, kwspan = phore.get_holofunc_args(span)
                    inst = phore.invoke(Cls, '__init__', span.kargs, span.kwargs, optional=False)
                    phore.span_bindings[span.uuid] = inst

                    # TODO extract to call_class_init (formalizes the api)
                    inst = phore.invoke(inst, '__holo_init__', kspan, kwspan)
                    if inst:
                        phore.span_bindings[span.uuid] = inst

        logger.pop()

        # --- Context Management ---
        # Ensure there's an initial context if the holoware starts with content
        if self.spans and not isinstance(self.spans[0], ContextResetSpan):
            phore.new_context()

        # --- Lifecycle: Main ---
        logger.push_debug("(main)")
        for span in self.spans:
            if isinstance(span, (TextSpan, ObjSpan)):
                if isinstance(span, TextSpan):
                    phore.add_text(span.text)
                elif isinstance(span, ObjSpan):
                    for var_id in span.var_ids:
                        if var_id in phore.env:
                            value = phore.env[var_id]
                            phore.add_text(f"<obj id={var_id}>")
                            phore.add_text(str(value))
                            phore.add_text("</obj>")
                            break
                continue

            SpanClassName = type(span).__name__
            # Create a rich-formatted log message with color
            logger.debug(f"[{span.get_color()}]<|{span}|>[/]")
            logger.push()
            if SpanClassName in HolowareHandlers.__dict__:
                getattr(HolowareHandlers, SpanClassName)(phore, span)
            else:
                logger.error(f"Could not find handler in HolowareHandlers for {SpanClassName}")

            if phore._newtext:
                logger.info(Text("" + phore._newtext, style="dim italic")) # TODO if we could find a 'oduble dim' color that would be better
                phore._newtext = ""
            logger.pop()

        if phore.errors > 0:
            raise RuntimeError(f"Failed to instantiate {phore.errors} classes.")
        logger.pop()


        # --- Lifecycle: End ---
        logger.push_debug("(end)")
        logger.debug("Span Bindings:")
        logger.debug(phore.span_bindings)
        for uid, target in phore.span_bindings.items():
            if hasattr(target, '__holo_end__'):
                span = phore.find_span(uid)
                # TODO extract to call_class_init (formalizes the api)
                kspan, kwspan = phore.get_holofunc_args(span)
                phore.invoke(target, '__holo_end__', kspan, kwspan)
        logger.pop()


        phore.active_holowares.remove(self)

        logger.pop()
        return phore


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
        ret = cls.parse(content)
        ret.name = Path(filepath).name
        return ret

    def __repr__(self):
        return f"Holoware({self.filepath})"

    def _append_args_to_debug(self, out: Text, span: Span):
        """Append formatted var_args and var_kwargs to the rich text."""
        args_text = []
        if span.kargs:
            args_text.append(", ".join(map(str, span.kargs)))
        if span.kwargs:
            args_text.append(", ".join(f"{k}={v}" for k, v in span.kwargs.items()))

        if args_text:
            out.append(" | ", style="dim white")
            out.append(", ".join(args_text), style="magenta")

    def to_rich(self, idt=0) -> Text:
        """Format the prompt template spans for rich debug display."""
        out = Text()
        if not self.spans:
            return out

        if idt == 0:
            # Create a fancy header with prompt info
            header = Text()
            header.append("╔══ ", style="bright_black")
            header.append("Holoware Template ", style="bold cyan")
            header.append("═" * 40, style="bright_black")
            header.append("\n")

            header.append("║ ", style="bright_black")
            header.append(f"Spans: ", style="dim")
            header.append(f"{len(self.spans)}", style="cyan")
            header.append(" | ", style="dim")
            header.append("Training Contexts: ", style="dim")
            header.append(f"{len(self.trained_contexts)}", style="cyan")
            header.append("\n")

            out.append(header)

        idt_text = "  " * idt

        # Calculate the width needed for span indices
        imax = len(self.spans) - 1
        iwidth = len(str(imax))
        ifmt = f"[{{:{iwidth}d}}] "
        iprocessed = set()

        for i, span in enumerate(self.spans):
            if i in iprocessed:
                continue

            out.append(idt_text)
            out.append(ifmt.format(i), style="dim white")

            if isinstance(span, ContextResetSpan):
                c = span.get_color()
                out.append("ContextResetSpan", style=c)
                if span.train:
                    out.append(" (train=True)", style=c)
                self._append_args_to_debug(out, span)
                out.append("\n")

            elif isinstance(span, EgoSpan):
                c = span.get_color()
                out.append("RoleSpan (", style=c)
                role_display = span.ego.replace('\n', '\\n').replace('\r', '\\r')
                out.append(f"role={role_display}", style=c)
                self._append_args_to_debug(out, span)
                out.append(")", style=c)

                # Look ahead - if next span is a single TextSpan, combine them for compact view
                inext = i + 1
                if inext < len(self.spans) and isinstance(self.spans[inext], TextSpan):
                    next = self.spans[inext]
                    next_after = inext + 1
                    next_after = self.spans[next_after] if next_after < len(self.spans) else None

                    # Pyright doesn't understand the type of next_span here, so we assert it
                    assert isinstance(next, TextSpan)

                    if not next_after or isinstance(next_after, (EgoSpan, ContextResetSpan)):
                        out.append(" → ", style="dim white")
                        out.append(f"'{next.display_text}{'...' if len(next.text) > 30 else ''}'", style="white")
                        self._append_args_to_debug(out, next)
                        iprocessed.add(inext)

                out.append("\n")

            elif isinstance(span, SampleSpan):
                c = span.get_color()
                out.append("SamplerSpan (", style=c)
                segs = []
                if span.uuid:
                    segs.append(f"id={span.uuid}")
                if span.goal:
                    segs.append(f"goal={span.display_goal}...")
                if span.fence:
                    fence_display = span.fence.replace('\n', '\\n').replace('\r', '\\r')
                    segs.append(f"fence={fence_display}")
                out.append(", ".join(segs), style=c)
                self._append_args_to_debug(out, span)
                out.append(")\n", style=c)

            elif isinstance(span, TextSpan):
                c = span.get_color()
                out.append(f"TextSpan ('{span.display_text}{'...' if len(span.text) > 30 else ''}')", style=c)
                self._append_args_to_debug(out, span)
                out.append("\n")

            elif isinstance(span, ObjSpan):
                c = span.get_color()
                out.append("ObjSpan (", style=c)
                out.append(f"vars={span.var_ids}", style=c)
                self._append_args_to_debug(out, span)
                out.append(")\n", style=c)

            elif isinstance(span, ClassSpan):
                c = span.get_color()
                out.append("ClassSpan (", style=c)
                out.append(f"class={span.class_name}", style=c)
                self._append_args_to_debug(out, span)
                out.append(")", style=c)
                if span.body:
                    out.append("\n")
                    out.append(span.body.to_rich(idt + 1))
                else:
                    out.append("\n")

        if idt == 0:
            out.append("╚", style="bright_black")
            out.append("═" * 60, style="bright_black")
            out.append("\n\n")

        return out

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
