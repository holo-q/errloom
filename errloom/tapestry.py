from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union, Literal
import re
from enum import Enum

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule

from errloom.utils.openai_chat import MessageList, MessageTuple

class FragmentType(Enum):
    """Type of text fragment for training purposes."""
    FROZEN = "frozen"          # Input text, typically masked
    REINFORCE = "reinforced"  # Text to reinforce (unmasked)

@dataclass
class TextFragment:
    """
    A fragment of text with training metadata.
    """
    content: str
    role: str
    fragment_type: FragmentType
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Context:
    """
    Represents a conversation context with text fragments and training metadata.
    TODO instead of _bake every time we should invalidate, make the baked fields private, and expose them with properties that cause bake if invalidated
    """
    text: str = ""
    fragments: List[TextFragment] = field(default_factory=list)
    mode: Literal["chat", "completion"] = "chat"

    _messages: MessageList = field(default_factory=list)

    def add_fragment(self, content: str, fragment_type: FragmentType, role: Optional[str], **metadata):
        """Add a text fragment with training metadata."""
        fragment = TextFragment(
            content=content,
            fragment_type=fragment_type,
            role=role,
            metadata=metadata
        )
        self.fragments.append(fragment)

        # Update legacy structures based on mode
        if self.mode == "chat" and role:
            self._update_messages_from_fragments()
        else:
            self._update_text_from_fragments()

    # TODO we need to redo this API to be always explicit about what is frozen and what isn't. No backwards compatibility, we just build the perfect API here and let errors domino propagate

    def add_text(self, text: str):
        """Legacy method - adds as masked prompt by default."""
        if self._messages and self._messages[-1].get('role') == 'assistant':
            # Continuing assistant message
            self.add_fragment(text, FragmentType.REINFORCE, role='assistant')
        else:
            self.add_fragment(text, FragmentType.FROZEN, role=None)

    def add_message(self, ego: str, text: str):
        """Legacy method - all assistant output is reinforced."""
        fragment_type = FragmentType.REINFORCE if ego == 'assistant' else FragmentType.FROZEN
        self.add_fragment(text, fragment_type, role=ego)

    def add_frozen(self, role: Optional[str], content: str):
        """Add text to mask (ignored in training)."""
        self.add_fragment(content, FragmentType.FROZEN, role=role)

    def add_reinforced(self, role: Optional[str], content: str):
        """Add text to reinforce (unmasked in training)."""
        self.add_fragment(content, FragmentType.REINFORCE, role=role)

    def _update_messages_from_fragments(self):
        """Update _messages from fragments for chat mode."""
        self._messages.clear()
        current_role = None
        current_content = ""

        for fragment in self.fragments:
            if fragment.role != current_role:
                # Role change - finish previous message
                if current_role and current_content:
                    self._messages.append({'role': current_role, 'content': current_content})
                current_role = fragment.role
                current_content = fragment.content
            else:
                # Same role - accumulate content
                current_content += fragment.content

        # Add final message
        if current_role and current_content:
            self._messages.append({'role': current_role, 'content': current_content})

    def _update_text_from_fragments(self):
        """Update text from fragments for completion mode."""
        self.text = "".join(fragment.content for fragment in self.fragments)

    def get_training_data(self) -> Dict[str, List[Any]]:
        """
        Extract training data with fine-grained fragment control.

        Returns:
            Dict with fragment_contents, fragment_types, fragment_roles, masks
        """
        return {
            'fragment_contents': [f.content for f in self.fragments],
            'fragment_types': [f.fragment_type.value for f in self.fragments],
            'fragment_roles': [f.role for f in self.fragments],
            'masks': [f.fragment_type in [FragmentType.REINFORCE, FragmentType.REINFORCE] for f in self.fragments]
        }

    @property
    def messages(self) -> MessageTuple:
        """Get messages for chat mode."""
        if self.mode != "chat":
            raise ValueError("Messages only available in chat mode")
        return tuple(self._messages)

    def to_openai_messages(self) -> List[Dict[str, str]]:
        """
        Convert context messages to OpenAI API format.

        Returns:
            List of message dicts compatible with OpenAI chat completions API.
        """
        if self.mode != "chat":
            raise ValueError("OpenAI messages only available in chat mode")

        api_messages = []
        for msg in self._messages:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                api_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        return api_messages

    def to_completion_text(self) -> str:
        """
        Convert context to plain text format for completion API.

        Returns:
            Concatenated text content, suitable for completions API.
        """
        if self.mode == "completion":
            return self.text

        # For chat mode, concatenate message contents
        text_parts = []
        for msg in self._messages:
            content = msg.get('content', '')
            if content:
                text_parts.append(content)
        return ''.join(text_parts)

    def set_mode(self, mode: Literal["chat", "completion"]):
        """Set the context mode and update internal structures."""
        self.mode = mode
        if mode == "chat":
            self._update_messages_from_fragments()
        else:
            self._update_text_from_fragments()

    @staticmethod
    def from_openai_messages(openai_messages: List[Dict[str, str]], text: str = "") -> 'Context':
        """
        Create a Context from OpenAI API format messages.

        Args:
            openai_messages: List of OpenAI format message dicts
            text: Optional text field to set

        Returns:
            New Context instance
        """
        context = Context(text=text)
        for msg in openai_messages:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                context.add_message(msg['role'], msg['content'])
        return context

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

        if not wrapper_tag:
            # Return last message content if no tag specified
            for msg in reversed(self._messages):
                if role is None or msg.get("role") == role:
                    return msg.get("content", "").strip()
            return None

        tag = wrapper_tag.lower()

        for msg in reversed(self._messages):
            if role is not None and msg.get("role") != role:
                continue
            content = msg.get("content", "")
            matches = list(re.finditer(fr'<{tag}>\s*(.*?)\s*(?:</{tag}>|$)', content, re.DOTALL))
            if matches:
                return matches[-1].group(1).strip()

        return None

    def extract_json(self, role: str = 'assistant') -> Optional[str]:
        """
        Extract JSON from context messages.

        Args:
            role: Role to search within (default: 'assistant')

        Returns:
            Extracted JSON string or None if not found
        """
        import re
        from errloom.utils import log

        # Find the last message from the specified role
        content = None
        for msg in reversed(self._messages):
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
    def extract_rollout_data(rollout: 'Rollout') -> Dict[str, Any]:
        """
        Extract standardized data from a rollout for dataset creation.

        Args:
            rollout: The rollout to extract data from

        Returns:
            Dictionary containing the extracted data fields
        """
        # Get completion from last sample message or fallback to empty string
        completion = ''
        if rollout.samples:
            last_sample = rollout.samples[-1]
            completion = last_sample.get('content', '') if isinstance(last_sample, dict) else str(last_sample)

        return {
            'prompt': rollout.row.get('prompt', ''),
            'completion': completion,
            'answer': rollout.row.get('answer', ''),
            'gravity': rollout.gravity,
            'task': rollout.task
        }

    @staticmethod
    def extract_rollouts_to_dataset(rollouts: List['Rollout'],
                                   state_columns: List[str] = [],
                                   extra_columns: List[str] = []) -> Dict[str, List[Any]]:
        """
        Extract data from a list of rollouts to create a dataset dictionary.

        Args:
            rollouts: List of rollouts to extract data from
            state_columns: Additional columns to extract from rollout.extra
            extra_columns: Additional columns to extract from rollout.extra

        Returns:
            Dictionary with lists of values for each column
        """
        # Initialize base data structure
        data_dict = {
            'prompt': [],
            'completion': [],
            'answer': [],
            'gravity': []
        }

        # Add task column if any rollout has non-default task
        has_custom_task = any(rollout.task != "default" for rollout in rollouts)
        if has_custom_task:
            data_dict['task'] = []

        # Initialize additional columns
        all_extra_cols = set(state_columns + extra_columns)
        for col in all_extra_cols:
            data_dict[col] = []

        # Extract data from each rollout
        for rollout in rollouts:
            rollout_data = Context.extract_rollout_data(rollout)

            # Add base fields
            data_dict['prompt'].append(rollout_data['prompt'])
            data_dict['completion'].append(rollout_data['completion'])
            data_dict['answer'].append(rollout_data['answer'])
            data_dict['gravity'].append(rollout_data['gravity'])

            # Add task if needed
            if has_custom_task:
                data_dict['task'].append(rollout_data['task'])

            # Add extra columns
            for col in all_extra_cols:
                data_dict[col].append(rollout.extra.get(col, None))

        return data_dict

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
    Fragment-based architecture for fine-grained training control.
    """
    row: Dict[str, Any]
    contexts: list[Context] = field(default_factory=list)
    gravity: float = 0.0
    gravities: Dict[str, float] = field(default_factory=dict)
    sampling_args: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    task: str = 'default'

    # Legacy compatibility fields (auto-computed from fragments)
    mask: list[bool] = field(default_factory=list, init=False)
    samples: list[Dict[str, str]] = field(default_factory=list, init=False)

    def __post_init__(self):
        pass

    def new_context(self, mode: Literal["chat", "completion"] = "chat"):
        """Create a new context with specified mode."""
        context = Context(mode=mode)
        self.contexts.append(context)

    @property
    def active_context(self) -> Context:
        """Get the currently active context."""
        if not self.contexts:
            raise RuntimeError("No contexts available - call new_context() first")
        return self.contexts[-1]

    def add_text(self, text: str):
        """Legacy method - delegates to active context."""
        self.active_context.add_text(text)
        self._bake()

    def add_message(self, ego: str, text: str):
        """Legacy method - delegates to active context."""
        self.active_context.add_message(ego, text)
        self._bake()

    # New fragment-based API
    def add_prompt(self, content: str, role: Optional[str]):
        """Add prompt text (typically masked)."""
        self.active_context.add_prompt(role, content)
        self._bake()

    def add_completion(self, content: str, role: Optional[str]):
        """Add completion text (typically reinforced)."""
        self.active_context.add_completion(role, content)
        self._bake()

    def add_reinforced(self, content: str, role: Optional[str]):
        """Add text to reinforce (unmasked in training)."""
        self.active_context.add_reinforced(role, content)
        self._bake()

    def add_masked(self, content: str, role: Optional[str]):
        """Add text to mask (ignored in training)."""
        self.active_context.add_frozen(role, content)
        self._bake()

    def set_mode(self, mode: Literal["chat", "completion"]):
        """Set the mode for all contexts."""
        for context in self.contexts:
            context.set_mode(mode)
        self._bake()

    def _bake(self):
        """Bake mask/samples fields from fragment data."""
        if not self.contexts:
            return

        # Get training data from active context
        training_data = self.active_context.get_training_data()

        self.mask = training_data['masks']
        self.samples = []
        contents = training_data['fragment_contents']
        roles = training_data['fragment_roles']

        for content, role in zip(contents, roles):
            if role:  # Only add fragments with roles as samples
                self.samples.append({'role': role, 'content': content})

    def get_sample_messages(self) -> List[Dict[str, str]]:
        """Get samples as properly formatted message objects."""
        try:
            result = []
            for i, msg in enumerate(self.samples):
                if isinstance(msg, dict):
                    result.append(dict(msg))  # Return clean copies
                else:
                    # Handle malformed samples gracefully
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Sample {i} is not a dict: {type(msg)} = {msg!r}")
                    # Try to convert to proper format
                    if isinstance(msg, str):
                        result.append({'role': 'assistant', 'content': msg})
                    else:
                        # Skip malformed entries
                        logger.error(f"Skipping malformed sample {i}: {msg!r}")
            return result
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in get_sample_messages: {e}")
            logger.error(f"self.samples = {self.samples!r}")
            # Return empty list to prevent crash
            return []

    def get_sample_strings(self) -> List[str]:
        """Get samples as strings for backward compatibility."""
        return [msg.get("content", "") for msg in self.samples]

    def get_mask_for_samples(self) -> List[bool]:
        """Get mask array corresponding to samples."""
        return self.mask.copy()

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
        """Legacy property - use active_context instead."""
        return self.active_context

    def to_openai_messages(self) -> List[Dict[str, str]]:
        """
        Get OpenAI API format messages from the current context.

        Returns:
            List of message dicts compatible with OpenAI chat completions API.
        """
        return self.context.to_openai_messages()

    def to_completion_text(self) -> str:
        """
        Get completion text from the current context.

        Returns:
            Text suitable for OpenAI completions API.
        """
        return self.context.to_completion_text()

    @staticmethod
    def from_openai_messages(openai_messages: List[Dict[str, str]], row: Dict[str, Any], **kwargs) -> 'Rollout':
        """
        Create a Rollout from OpenAI API format messages.

        Args:
            openai_messages: List of OpenAI format message dicts
            row: Initial row data for the rollout
            **kwargs: Additional rollout parameters

        Returns:
            New Rollout instance with a context populated from the messages
        """
        rollout = Rollout(row=row, **kwargs)
        context = Context.from_openai_messages(openai_messages)
        rollout.contexts.append(context)
        return rollout

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
        return self.context.extract_json(role)

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
        return Context.extract_rollouts_to_dataset(
            self.rollouts,
            state_columns=state_columns,
            extra_columns=extra_columns
        )