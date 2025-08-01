from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule

from errloom.aliases import APIChat

class FragmentType(Enum):
    """Type of text fragment for training purposes."""
    FROZEN = "frozen"  # Input text, typically masked
    REINFORCE = "reinforced"  # Text to reinforce (unmasked)

@dataclass
class TextFragment:
    """
    A fragment of text with training metadata.
    """
    content: str
    ego: Optional[str]
    fragment_type: FragmentType
    metadata: Dict[str, Any] = field(default_factory=dict)

class AutoMask(Enum):
    FREEZE_ALL = 0
    REINFORCE_ALL = 1
    REINFORCE_USER = 2
    REINFORCE_ASSISTANT = 3

@dataclass
class Context:
    """
    Represents a conversation context with text fragments and training metadata.
    TODO maybe invalidatation and auto-bake with cached backing field, auto baked when messages or text accessed by property
    """
    text: str = ""
    fragments: List[TextFragment] = field(default_factory=list)

    def add_fragment(self, content: str, fragment_type: FragmentType, role: Optional[str], **metadata):
        """Add a text fragment with training metadata."""
        self.fragments.append(TextFragment(content=content,
            fragment_type=fragment_type,
            ego=role,
            metadata=metadata
        ))

    # def add_text(self, text: str):
    #     """Legacy method - adds as masked prompt by default."""
    #     if self._messages and self._messages[-1].get('role') == 'assistant':
    #         # Continuing assistant message
    #         self.add_fragment(text, FragmentType.REINFORCE, role='assistant')
    #     else:
    #         self.add_fragment(text, FragmentType.FROZEN, role=None)
    #
    # def add_message(self, ego: str, text: str):
    #     """Legacy method - all assistant output is reinforced."""
    #     fragment_type = FragmentType.REINFORCE if ego == 'assistant' else FragmentType.FROZEN
    #     self.add_fragment(text, fragment_type, role=ego)

    def add_frozen(self, role: Optional[str], content: str):
        """Add text to mask (ignored in training)."""
        self.add_fragment(content, FragmentType.FROZEN, role=role)

    def add_reinforced(self, role: Optional[str], content: str):
        """Add text to reinforce (unmasked in training)."""
        self.add_fragment(content, FragmentType.REINFORCE, role=role)

    def to_api_messages(self) -> APIChat:
        """
        Convert context fragments to OpenAI API chat messages by aggregating
        consecutive fragments with the same normalized role.
        """
        def _normalize_role(raw: Optional[str], is_first: bool) -> str:
            if raw in ("system", "user", "assistant"):
                return raw
            # Unknowns map to user; None maps to system only if first fragment
            if raw is None and is_first:
                return "system"
            return "user"

        messages: APIChat = []
        current_role: Optional[str] = None
        current_content: list[str] = []

        for idx, frag in enumerate(self.fragments):
            role = _normalize_role(frag.ego, is_first=(idx == 0))
            if current_role is None:
                current_role = role
                current_content = [frag.content]
            elif role == current_role:
                current_content.append(frag.content)
            else:
                # flush
                content_str = "".join(current_content)
                if content_str:
                    messages.append({"role": current_role, "content": content_str})
                current_role = role
                current_content = [frag.content]

        # flush tail
        if current_role is not None:
            content_str = "".join(current_content)
            if content_str:
                messages.append({"role": current_role, "content": content_str})

        return messages

    def to_api_string(self) -> str:
        """
        Convert context to plain text format for completion API using legacy
        delimiter blocks:
          <|im_start|>role
          content
          <|im_end|>
        """
        blocks: list[str] = []
        for msg in self.to_api_messages():
            role = msg.get("role", "user")
            content = msg.get("content", "")
            blocks.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
        return "\n".join(blocks)

    # def _update_messages_from_fragments(self):
    #     """Update _messages from fragments for chat mode."""
    #     self._messages.clear()
    #     current_role = None
    #     current_content = ""
    #
    #     for fragment in self.fragments:
    #         if fragment.role != current_role:
    #             # Role change - finish previous message
    #             if current_role and current_content:
    #                 self._messages.append({'role': current_role, 'content': current_content})
    #             current_role = fragment.role
    #             current_content = fragment.content
    #         else:
    #             # Same role - accumulate content
    #             current_content += fragment.content
    #
    #     # Add final message
    #     if current_role and current_content:
    #         self._messages.append({'role': current_role, 'content': current_content})
    #
    # def _update_text_from_fragments(self):
    #     """Update text from fragments for completion mode."""
    #     self.text = "".join(fragment.content for fragment in self.fragments)

    # def set_mode(self, mode: Literal["chat", "completion"]):
    #     """Set the context mode and update internal structures."""
    #     self.mode = mode
    #     if mode == "chat":
    #         self._update_messages_from_fragments()
    #     else:
    #         self._update_text_from_fragments()


    def extract_fence(self, wrapper_tag: Optional[str], role: str = 'assistant') -> Optional[str]:
        """
        Extract content from a dynamic wrapper tag, e.g., <compress>...</compress>

        Args:
            wrapper_tag: The tag name to extract content from
            role: Role to search within (default: 'assistant')

        Returns:
            Extracted content or None if not found
        """
        import re
        messages = self.to_api_messages()

        if not wrapper_tag:
            # Return last message content if no tag specified
            for msg in reversed(messages):
                if role is None or msg.get("role") == role:
                    return msg.get("content", "").strip()
            return None

        tag = wrapper_tag.lower()

        for msg in reversed(messages):
            if role is not None and msg.get("role") != role:
                continue
            content = msg.get("content", "")
            matches = list(re.finditer(fr'<{tag}>\s*(.*?)\s*(?:</{tag}>|$)', content, re.DOTALL))
            if matches:
                return matches[-1].group(1).strip()

        return None

    def extract_mdjson(self, role: str = 'assistant') -> Optional[str]:  # TODO return dict
        """
        Extract markdown JSON block from context messages, e.g.:

        ```json
        {
            "key": "value"
        }
        ```

        returns dict(key="value")


        Args:
            role: Role to search within (default: 'assistant')

        Returns:
            Extracted JSON string or None if not found
        """
        import re
        from errloom.lib import log
        messages = self.to_api_messages()

        # Find the last message from the specified role
        content = None
        for msg in reversed(messages):
            if msg.get("role") == role:
                content = msg.get("content", "")
                break

        if not content:
            return None

        # Look for ```json blocks first
        block_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if block_match:
            return block_match.group(1).strip()

        # Look for standalone JSON object
        match = re.search(r'\{[^{}]*(?:\{[^{}]*}[^{}]*)*}', content, re.DOTALL)
        if match:
            return match.group(0)

        log.getLogger(__name__).warning("No JSON found in context")
        return None


    @staticmethod
    def from_api_chat(api_context: APIChat, text: str = "", masking=AutoMask.FREEZE_ALL) -> 'Context':
        """
        Create a Context from OpenAI API format messages, applying AutoMask.
        """
        def mask_for(role: Optional[str]) -> FragmentType:
            if masking == AutoMask.FREEZE_ALL:
                return FragmentType.FROZEN
            if masking == AutoMask.REINFORCE_ALL:
                return FragmentType.REINFORCE
            if masking == AutoMask.REINFORCE_USER:
                return FragmentType.REINFORCE if role == "user" else FragmentType.FROZEN
            if masking == AutoMask.REINFORCE_ASSISTANT:
                return FragmentType.REINFORCE if role == "assistant" else FragmentType.FROZEN
            return FragmentType.FROZEN

        context = Context(text=text)
        for msg in api_context:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content", "")
            if content is None:
                content = ""
            ftype = mask_for(role)
            context.add_fragment(content=content, fragment_type=ftype, role=role)
        return context

    @staticmethod
    def from_text(text: str, masking: AutoMask = AutoMask.FREEZE_ALL) -> 'Context':
        """
        Parse a conversation string using legacy delimiters into a Context.
        Format per block:
          <|im_start|>role
          content
          <|im_end|>
        """
        pattern = re.compile(
            r"<\|im_start\|\>(?P<role>[^\n\r]+)[\r]?\n(?P<content>.*?)[\r]?\n<\|im_end\|\>",
            re.DOTALL,
        )
        matches = list(pattern.finditer(text))
        if not matches:
            raise ValueError("from_text: no delimited messages found")

        ctx = Context()
        # Reuse from_api_chat masking by constructing APIChat then converting
        api_msgs: APIChat = []
        for m in matches:
            role = m.group("role").strip()
            content = m.group("content")
            api_msgs.append({"role": role, "content": content})
        return Context.from_api_chat(api_msgs, masking=masking)

    @property
    def text_rich(self):
        """Rich text representation of the context, with colored roles."""
        from rich.text import Text

        # Define color mapping for roles
        role_colors = {
            "system":    "bold cyan",
            "user":      "bold green",
            "assistant": "bold magenta",
            "unknown":   "dim",
        }

        messages = self.to_api_messages()
        text = Text()
        for msg in messages:
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
    Fragment-based architecture for fine-grained training control.
    """
    row: Dict[str, Any]
    contexts: list[Context] = field(default_factory=list)
    gravity: float = 0.0
    gravities: Dict[str, float] = field(default_factory=dict)
    sampling_args: Dict[str, Any] = field(default_factory=dict)

    # task: str = 'default'
    # extra: Dict[str, Any] = field(default_factory=dict)

    # Legacy compatibility fields (auto-computed from fragments)
    # mask: list[bool] = field(default_factory=list, init=False)
    # samples: list[Dict[str, str]] = field(default_factory=list, init=False)

    @property
    def active_context(self) -> Context:
        """Get the currently active context."""
        if not self.contexts:
            raise RuntimeError("No contexts available - call new_context() first")
        return self.contexts[-1]

    def new_context(self):
        """Create a new context with specified mode."""
        self.contexts.append(Context())

    def add_reinforced(self, content: str, role: Optional[str]):
        """Add text to reinforce (unmasked in training)."""
        self.active_context.add_reinforced(role, content)

    def add_frozen(self, content: str, role: Optional[str]):
        """Add text to mask (ignored in training)."""
        self.active_context.add_frozen(role, content)

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

    @property
    def context(self) -> Context:
        """Legacy property - use active_context instead."""
        return self.active_context

    def to_api_chat(self) -> APIChat:
        """
        Get OpenAI API format messages from the current context.

        Returns:
            List of message dicts compatible with "OpenAI" chat completions API.
        """
        return self.context.to_api_messages()

    def to_text(self) -> str:
        """
        Get text from the current context.
        """
        return self.context.to_api_string()

    def extract_fence(self, wrapper_tag: Optional[str], role: str = "assistant") -> Optional[str]:
        """
        Extract content from a dynamic wrapper tag across all contexts.

        Args:
            wrapper_tag: The tag name to extract content from
            role: Role to search within (default: 'assistant')

        Returns:
            Extracted content or None if not found
        """
        for context in self.contexts:
            result = context.extract_fence(wrapper_tag, role)
            if result:
                return result
        return None

    def extract_json(self, role: str = "assistant") -> Optional[str]:
        """
        Extract JSON from the current context.

        Args:
            role: Role to search within (default: 'assistant')

        Returns:
            Extracted JSON string or None if not found
        """
        return self.context.extract_mdjson(role)

    def to_rich(self) -> Panel:
        renderables = []
        for i, ctx in enumerate(self.contexts):
            if i > 0:
                renderables.append(Rule(style="yellow"))
            renderables.append(ctx.text_rich)
        group = Group(*renderables)

        panel = Panel(
            group,
            title="[bold yellow]Dry Run: Full Conversation Flow[/]",
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
