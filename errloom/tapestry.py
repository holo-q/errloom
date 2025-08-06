from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule

from errloom.aliases import APIChat
from errloom.context import Context, Frag, FragList, FragType
from errloom.lib import log

logger = log.getLogger(__name__)

@dataclass
class Rollout:
    """
    Work state for a rollout and final returned output.
    Fragment-based architecture for fine-grained training control.
    """
    row: Dict[str, Any]
    contexts: list[Context] = field(default_factory=list)
    gravity: float = 0.0
    gravities: Dict[str, float] = field(default_factory=dict)
    sampling_args: Dict[str, Any] = field(default_factory=dict)
    samples: list[Dict[str, str]] = field(default_factory=list, init=False)

    # task: str = 'default'
    # extra: Dict[str, Any] = field(default_factory=dict)

    # Legacy compatibility fields (auto-computed from fragments)
    # mask: list[bool] = field(default_factory=list, init=False)

    @property
    def active_context(self) -> Context:
        """Get the currently active context."""
        if not self.contexts:
            raise RuntimeError("No contexts available - call new_context() first")
        return self.contexts[-1]

    def new_context(self) -> Context:
        ret = Context()
        self.contexts.append(ret)
        return ret

    def ensure_context(self):
        if len(self.contexts) == 0:
            self.new_context()

    def add_reinforced(self, ego: Optional[str], content: str) -> Frag:
        """Add text to reinforce (unmasked in training)."""
        return self.active_context.add_reinforced(ego, content)

    def add_frozen(self, ego: Optional[str], content: str) -> Frag:
        """Add text to mask (ignored in training)."""
        return self.active_context.add_frozen(ego, content)

    def add_frag(self,
                 ego: Optional[str],
                 type: FragType,
                 content: str,
                 prints: bool = True) -> Frag:
        """Add a text fragment with training metadata."""
        self.ensure_context()
        return self.active_context.add_frag(ego, content, type, prints)

    # def bake_tokens(self):
    #     """Bake mask/samples fields from fragment data."""
    #     if not self.contexts:
    #         return
    #
    #     # Get training data from active context
    #     training_data = self.active_context.get_training_data()
    #
    #     self.mask = training_data['masks']
    #     self.samples = []
    #     contents = training_data['fragment_contents']
    #     roles = training_data['fragment_roles']
    #
    #     for content, role in zip(contents, roles):
    #         if role:  # Only add fragments with roles as samples
    #             self.samples.append({'role': role, 'content': content})


    # def get_sample_strings(self) -> List[str]:
    #     """Get samples as strings for backward compatibility."""
    #     return [msg.get("content", "") for msg in self.samples]
    #
    # def get_mask_for_samples(self) -> List[bool]:
    #     """Get mask array corresponding to samples."""
    #     return self.mask.copy()

    def __rich_repr__(self):
        yield "row", self.row
        yield "gravity", self.gravity
        yield "gravities", self.gravities

    def __repr__(self) -> str:
        from errloom.lib.log import PrintedText
        return str(PrintedText(self))

    def to_api_chat(self) -> APIChat:
        """
        Get OpenAI API format messages from the current context.

        Returns:
            List of message dicts compatible with "OpenAI" chat completions API.
        """
        return self.active_context.to_api_messages()

    def to_text(self) -> str:
        """
        Get text from the current context.
        """
        return self.active_context.to_api_string()

    def extract_fence(self, wrapper_tag: Optional[str], ego: str = "assistant") -> Optional[str]:
        """
        Extract content from a dynamic wrapper tag across all contexts.

        Args:
            wrapper_tag: The tag name to extract content from
            ego: Ego to search within (default: 'assistant')

        Returns:
            Extracted content or None if not found
        """
        for context in self.contexts:
            result = context.extract_xml_tag(wrapper_tag, ego)
            if result:
                return result
        return None

    def extract_json(self, ego: str = "assistant") -> Optional[str]:
        """
        Extract JSON from the current context.

        Args:
            ego: Ego to search within (default: 'assistant')

        Returns:
            Extracted JSON string or None if not found
        """
        return self.active_context.extract_markdown_json(ego)

    def to_rich(self) -> Panel:
        out = []
        for i, ctx in enumerate(self.contexts):
            if i > 0:
                out.append(Rule(style="yellow"))

            # Build a display-only transcript using the API-chat path to ensure rich rendering
            msgs = ctx.to_api_messages(render_dry=True)

            # If nothing made it through (e.g. all masked/empty), provide a placeholder to avoid blank panel
            if not msgs:
                tmp_ctx = Context()
                tmp_ctx.add_frag(ego="system", text="[empty context]", type=FragType.FROZEN)
                out.append(tmp_ctx.to_rich())
                continue

            out.append(ctx.to_rich())

        group = Group(*out)
        panel = Panel(
            group,
            title="[bold yellow]Rollout.to_rich()[/]",
            border_style="yellow",
            box=box.ROUNDED,
            title_align="left"
        )

        return panel

    @staticmethod
    def from_api_chat(api_chat: APIChat, **kwargs) -> 'Rollout':
        """
        Create a Rollout from OpenAI API format messages.

        Args:
            api_chat: List of OpenAI format message dicts
            **kwargs: Additional rollout parameters

        Returns:
            New Rollout instance with a context populated from the messages
        """
        rollout = Rollout(**kwargs)
        context = Context.from_api_chat(api_chat)
        rollout.contexts.append(context)
        return rollout

    @staticmethod
    def from_text(conversation: str):
        raise NotImplementedError  # TODO parse the <|im_start|>user: and stuff into fragment list

@dataclass
class Tapestry:
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

    def to_dataset(self, state_columns: List[str] = [], extra_columns: List[str] = []) -> Dict[str, List[Any]]:
        """
        Convert the tapestry to a dataset dictionary.

        Args:
            state_columns: Additional columns to extract from rollout.extra
            extra_columns: Additional columns to extract from rollout.extra

        Returns:
            Dictionary with lists of values for each column
        """
        return self.extract_rollouts_to_dataset(state_columns=state_columns, extra_columns=extra_columns)

    def extract_rollouts_to_dataset(self,
                                    state_columns: List[str] = [],
                                    extra_columns: List[str] = []) -> Dict[str, List[Any]]:
        """
        Extract minimal dataset fields from rollouts.
        - prompt: concatenation of all non-assistant messages as "role: content" separated by blank lines
        - completion: content of the last assistant message if any, else ""
        - answer: alias of completion
        - gravity: rollout.gravity
        """
        data_dict: Dict[str, List[Any]] = {
            'prompt':     [],
            'completion': [],
            'answer':     [],
            'gravity':    []
        }

        for rollout in self.rollouts:
            msgs = rollout.to_api_chat()
            # Build prompt from non-assistant messages
            prompt_parts: list[str] = []
            last_assistant: str = ""
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "assistant":
                    last_assistant = content  # overwrite to last assistant
                else:
                    prompt_parts.append(f"{role}: {content}")
            prompt = "\n\n".join(prompt_parts)
            completion = last_assistant
            answer = last_assistant

            data_dict['prompt'].append(prompt)
            data_dict['completion'].append(completion)
            data_dict['answer'].append(answer)
            data_dict['gravity'].append(getattr(rollout, "gravity", 0.0))

        return data_dict

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
