from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from errloom.holoware.openai_chat import MessageList

@dataclass
class Context:
    """
    Represents a conversation context with messages and related data.
    """
    text: str = ""
    messages: MessageList = field(default_factory=list)
    attractors: List[Any] = field(default_factory=list)

    @property
    def text_rich(self):
        """Rich text representation of the context."""
        from rich.text import Text
        text = Text()
        for msg in self.messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            text.append(f"{role}: {content}\n")
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

    def __repr__(self) -> str:  # noqa: D401
        lines = [
            f"Rollout(",
            f"  • row: {_truncate_long_strings(self.row)}",
            f"  • samples: {_truncate_long_strings(self.samples)}",
            f"  • gravity: {self.gravity}",
            f"  • gravities: {_truncate_long_strings(self.gravities)}",
            f"  • extra: {_truncate_long_strings(self.extra)}",
            f")"
        ]
        return "\n".join(lines)


    @property
    def context(self) -> Context:
        return self.contexts[-1]

@dataclass
class Rollouts:
    """
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