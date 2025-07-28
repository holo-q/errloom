from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import re

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule

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

    def to_openai_messages(self) -> List[Dict[str, str]]:
        """
        Convert context messages to OpenAI API format.
        
        Returns:
            List of message dicts with 'role' and 'content' keys,
            compatible with OpenAI chat completions API.
        """
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
            Concatenated text content from all messages, suitable for
            OpenAI completions API.
        """
        if self.text:
            return self.text
        
        text_parts = []
        for msg in self._messages:
            content = msg.get('content', '')
            if content:
                text_parts.append(content)
        return ''.join(text_parts)

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
        return {
            'prompt': rollout.row.get('prompt', ''),
            'completion': rollout.samples[-1] if rollout.samples else '',
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
    Gravity work also occurs in this state and mutates it.
    # TODO Possible rename to String, Lstring, Wstring or Unstring. Treating it as a super-string is a very interesting continuation of the software engineering semantic.
    """
    row: Dict[str, Any]
    contexts: list[Context] = field(default_factory=list)
    mask: list[bool] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)
    gravity: float = 0.0
    gravities: Dict[str, float] = field(default_factory=dict)
    sampling_args: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    task: str = 'default'



    def __post_init__(self):
        pass

    def new_context(self):
        self.contexts.append(Context())

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