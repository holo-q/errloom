from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List
import re

from errloom.utils.openai_chat import MessageList, MessageTuple

@dataclass
class Context:
    """
    Represents a conversation context with messages and related data.
    """
    text: str = ""
    _messages: MessageList = field(default_factory=list)
    # attractors: List[Any] = field(default_factory=list)

    def add_text(self, text: str):
        self._messages[-1]['content'] += text

    def add_message(self, ego: str, text: str):
        self._messages.append({'role': ego, 'content': text})

    @property
    def messages(self) -> MessageTuple:
        return tuple(self._messages)

    @property
    def text_rich(self):
        """Rich text representation of the context, with colored roles."""
        from rich.text import Text

        # Define color mapping for roles
        role_colors = {
            "system": "bold cyan",
            "user": "bold green",
            "assistant": "bold magenta",
            "unknown": "dim",
        }

        text = Text()
        for msg in self.messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            color = role_colors.get(role, "white")
            text.append(f"{role}:", style=color)

            highlighted_content = Text(style="white")
            # Regex to find <tag>...</tag> and <tag id=foo>...</tag>
            # It will match <obj ...>, <compress>, <decompress>, <think>, <json>
            pattern = re.compile(r"(<(obj|compress|decompress|think|json|critique)\b[^>]*>.*?<\/\2>)", re.DOTALL)

            last_idx = 0
            for match in pattern.finditer(content):
                start, end = match.span()
                # Append text before the match
                highlighted_content.append(content[last_idx:start])

                full_match_text = match.group(1)
                tag_name = match.group(2)

                # Regex to parse the tag and content
                tag_pattern = re.compile(r"<(?P<tag_name>\w+)(?P<attributes>[^>]*)>(?P<inner_content>.*?)</\1>", re.DOTALL)
                tag_match = tag_pattern.match(full_match_text)

                if tag_match:
                    attributes_str = tag_match.group('attributes')
                    inner_content = tag_match.group('inner_content')

                    style = "yellow"
                    if tag_name in ["compress", "decompress", "think", "json", "critique"]:
                        style = "blue"

                    # Reconstruct and style
                    highlighted_content.append(f"<{tag_name}{attributes_str}>", style=f"bold {style}")
                    highlighted_content.append(inner_content, style=style)
                    highlighted_content.append(f"</{tag_name}>", style=f"bold {style}")
                else:
                    # Fallback for non-matching (should not happen with the outer regex)
                    highlighted_content.append(full_match_text)

                last_idx = end

            # Append any remaining text
            highlighted_content.append(content[last_idx:])

            text.append(f" ")
            text.append(highlighted_content)
            text.append(f"\n", style="white")

        return text

def _truncate_long_strings(data: Any, max_len: int = 128) -> Any:
    """Recursively truncate long strings in a data structure."""
    if isinstance(data, str):
        if len(data) > max_len:
            return data[:max_len] + "..."
        return data
    if isinstance(data, dict):
        return {k: _truncate_long_strings(v, max_len) for k, v in data.items()}
    if isinstance(data, list):
        return [_truncate_long_strings(v, max_len) for v in data]
    if isinstance(data, tuple):
        return tuple(_truncate_long_strings(v, max_len) for v in data)
    return data

def _pretty_repr(obj: Any) -> str:
    """Generate a pretty, multiline repr for a dataclass instance."""
    data = asdict(obj)
    truncated_data = _truncate_long_strings(data)
    formatted_data = json.dumps(truncated_data, indent=2)
    # Indent every line of the JSON string to make it look nested.
    indented_data = "\n".join(f"  {line}" for line in formatted_data.splitlines())
    return f"{obj.__class__.__name__}(\n{indented_data}\n)"

@dataclass
class Rollout:
    """
    Work state for a rollout and final returned output.
    Gravity work also occurs in this state and mutates it.
    """
    row: Dict[str, Any]
    contexts: list[Context] = field(default_factory=lambda: [Context()])
    mask: list[bool] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)
    gravity: float = 0.0
    gravities: Dict[str, float] = field(default_factory=dict)
    sampling_args: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    task: str = 'default'
    
    def add_text(self, text: str):
        self.contexts[-1].add_text(text)

    def add_message(self, ego: str, text: str):
        self.contexts[-1].add_message(ego, text)

    def __rich_repr__(self):
        yield "row", self.row
        yield "samples", self.samples
        yield "gravity", self.gravity
        yield "gravities", self.gravities
        yield "extra", self.extra

    def __repr__(self) -> str:
        from errloom.utils.log import PrintedText
        return str(PrintedText(self))

    @property
    def context(self) -> Context:
        return self.contexts[-1]

@dataclass
class Rollouts:
    """
    A tapestry of rollouts woven by a loom.
    Structured return type for ``Environment.generate``.

    This replaces the loosely–typed ``Dict[str, Any]`` contract previously used
    across the code-base.  All existing call-sites can continue to treat the
    object like a mapping (``__getitem__`` & ``keys`` helpers) while newer code
    can benefit from static typing.
    """
    rollouts: List[Rollout]
    task: str = "default"
    max_concurrent: int = 128

    def __repr__(self) -> str:  # noqa: D401
        return _pretty_repr(self)

    def __getitem__(self, index: int):
        return self.rollouts[index]

    def __iter__(self):
        return self.rollouts.__iter__()