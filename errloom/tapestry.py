from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule

from errloom.aliases import APIChat
from errloom.lib import log

logger = log.getLogger(__name__)

class FragType(Enum):
    """Type of text fragment for training purposes."""
    FROZEN = "frozen"  # Input text, typically masked
    REINFORCE = "reinforced"  # Text to reinforce (unmasked)

@dataclass
class Frag:
    """
    A fragment of text with training metadata.
    """
    text: str
    ego: Optional[str]
    type: FragType

    def __str__(self):
        return f"Frag({self.ego}->{log.ellipse(self.text, 50)})"

    def __repr__(self):
        return f"Frag({self.ego}->{log.ellipse(self.text, 50)})"

class FragList(list[Frag]):
    EMPTY: ClassVar[FragList] = None  # type: ignore

    @property
    def string(self):
        ret = ""
        for frag in self:
            ret += frag.text
        return ret

    @property
    def length(self):
        return len(self)

FragList.EMPTY = FragList()

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
    fragments: FragList = field(default_factory=FragList)

    def add_frag(self, ego: Optional[str], content: str, type: FragType, prints: bool = True) -> Frag:
        """Add a text fragment with training metadata."""
        frag = Frag(text=content, type=type, ego=ego)
        self.fragments.append(frag)
        if prints:
            logger.debug(f"add_fragment :: {ego} -> {log.ellipse(content, 45)}")
        return frag

    def add_frozen(self, role: Optional[str], content: str) -> Frag:
        """Add text to mask (ignored in training)."""
        ret = self.add_frag(role, content, FragType.FROZEN, prints=False)
        logger.debug(f"add_frozen :: [black on white]{role} -> {log.ellipse(content, 45)}[/]")
        return ret

    def add_reinforced(self, role: Optional[str], content: str) -> Frag:
        """Add text to reinforce (unmasked in training)."""
        ret = self.add_frag(role, content, FragType.REINFORCE, prints=False)
        logger.debug(f"add_reinforced: [black on white]{role} -> {log.ellipse(content, 45)}[/]")
        return ret

    def to_api_messages(self, render_dry: bool = False) -> APIChat:
        """
        Convert context fragments to OpenAI API chat messages by aggregating
        consecutive fragments with the same normalized role.
        Includes display-only sanitation to avoid role-label concatenation leaking into content.

        When render_dry is True, do not drop empty/whitespace-only messages so the dry-run
        view can show scaffolded messages that are masked during training.
        """

        def _normalize_role(raw: Optional[str], is_first: bool) -> str:
            # TODO print warnings
            if raw in ("system", "user", "assistant"):
                return raw
            if raw is None and is_first:
                return "system"
            return "user"

        messages: APIChat = []
        content: list[str] = []
        current_role: Optional[str] = None

        logger.debug(f"to_api_messages: Processing {len(self.fragments)} fragments")

        # TODO there may be some code here that we can move to FragList

        for i, frag in enumerate(self.fragments):
            normalized_role = _normalize_role(frag.ego, is_first=(i == 0))
            ellipsed_content = log.ellipse(frag.text.replace('\n', '\\n'), 45).strip()
            logger.debug(f"- Fragment {i}: [dim]{frag.ego}[/]->{normalized_role} :: [dim]{ellipsed_content}[/]")

            if current_role is None:
                # First fragment
                current_role = normalized_role
                content = [frag.text]
            elif normalized_role == current_role:
                # Same role, accumulate content
                content.append(frag.text)
            else:
                # Role changed, finalize previous message
                s = "".join(content)
                if s or render_dry:
                    messages.append({"role": current_role, "content": s.strip()})
                    # logger.debug(f"Role changed from {current_role} to {normalized_role}, added message with {len(s)} chars")
                current_role = normalized_role
                content = [frag.text]

        # flush tail
        if current_role is not None:
            s = "".join(content)
            if s or render_dry:
                messages.append({"role": current_role, "content": s})

        for i, msg in enumerate(messages):
            logger.debug(f"- Message {i}: role={msg.get('role')}, content_length={len(msg.get('content', ''))}")

        return messages

    def to_api_string(self) -> str:
        """
        Convert context to plain text format for completion API using legacy
        delimiter blocks:
          <|im_start|>role
          content
          <|im_end|>

        Special-case for completion-style prefixing:
        - If the last normalized fragment is assistant and content is empty,
          we end with an open assistant header to cue generation:
              ... <|im_end|>
              <|im_start|>assistant
        """
        messages = self.to_api_messages()
        blocks: list[str] = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            blocks.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")

        # TODO there may be some code here that we can move to FragList

        # Inspect raw fragments to decide on open assistant tail
        # Normalize last fragment role like in to_api_messages
        def _normalize_role(raw: Optional[str], is_first: bool) -> str:
            if raw in ("system", "user", "assistant"):
                return raw
            if raw is None and is_first:
                return "system"
            return "user"

        if self.fragments:
            # Find last fragment and its normalized role
            last_idx = len(self.fragments) - 1
            last_norm = _normalize_role(self.fragments[last_idx].ego, is_first=(last_idx == 0))

            # If trailing assistant fragment exists with empty content, open assistant header
            if last_norm == "assistant":
                # If the last assistant fragment has no content, signal open assistant turn.
                last_content = self.fragments[last_idx].text or ""
                if last_content == "":
                    suffix = "<|im_start|>assistant"
                    if blocks:
                        return "\n".join(blocks + [suffix])
                    else:
                        return suffix

        return "\n".join(blocks)

    def extract_xml_tag(self, tag: Optional[str], role: str = 'assistant') -> Optional[str]:
        """
        Extract content from a dynamic wrapper tag, e.g., <think>...</think>
        by searching backwards from the last message.

        Args:
            tag: The tag name to extract content from (e.g. think)
            role: Role to search within (default: 'assistant')

        Returns:
            Extracted content or None if not found
        """
        import re
        messages = self.to_api_messages()

        if not tag:
            # Return last message content if no tag specified
            for msg in reversed(messages):
                if role is None or msg.get("role") == role:
                    return msg.get("content", "").strip()
            return None

        t = tag.lower()

        for msg in reversed(messages):
            if role is not None and msg.get("role") != role:
                continue
            content = msg.get("content", "")
            matches = list(re.finditer(fr'<{t}>\s*(.*?)\s*(?:</{t}>|$)', content, re.DOTALL))
            if matches:
                return matches[-1].group(1).strip()

        return None

    def extract_markdown_json(self, role: str = 'assistant') -> Optional[str]:  # TODO return dict
        """
        Extract markdown JSON block from context messages
        by searching backwards from the last message, e.g.:

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

        def mask_for(role: Optional[str]) -> FragType:
            if masking == AutoMask.FREEZE_ALL:
                return FragType.FROZEN
            if masking == AutoMask.REINFORCE_ALL:
                return FragType.REINFORCE
            if masking == AutoMask.REINFORCE_USER:
                return FragType.REINFORCE if role == "user" else FragType.FROZEN
            if masking == AutoMask.REINFORCE_ASSISTANT:
                return FragType.REINFORCE if role == "assistant" else FragType.FROZEN
            return FragType.FROZEN

        context = Context(text=text)
        for msg in api_context:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content", "")
            if content is None:
                content = ""
            ftype = mask_for(role)
            context.add_frag(ego=role, content=content, type=ftype)
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

        # Reuse from_api_chat masking by constructing APIChat then converting
        msgs: APIChat = []
        for m in matches:
            role = m.group("role").strip()
            content = m.group("content")
            msgs.append({"role": role, "content": content})
        return Context.from_api_chat(msgs, masking=masking)

    def to_rich(self):
        """Rich text representation of the context, with colored roles."""
        from rich.text import Text
        import re as _re

        # Define color mapping for roles
        role_colors = {
            "system":    "bold cyan",
            "user":      "bold green",
            "assistant": "bold magenta",
            "unknown":   "dim",
        }

        messages = self.to_api_messages()
        ret = Text()

        for imsg, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            # Guard: ensure a known role for rendering
            if role not in ("system", "user", "assistant"):
                logger.error("Context.to_rich: Unknown role found in context: %s", role)
                role = "user"

            color = role_colors.get(role, "white")

            # Prepare highlighted content
            hlcontent = Text(style="white")
            # Regex to find <tag>...</tag> and <tag id=foo>...</tag>
            # It will match <obj ...>, <compress>, <decompress>, <think>, <json>, <critique>
            pattern = _re.compile(r"(<(obj|compress|decompress|think|json|critique)\b[^>]*>.*?<\/\2>)", _re.DOTALL)

            # Defensive cleanup (display-only): strip accidental role concatenations at content start
            sanitized_content = content
            if _re.match(r'^(?:system|user|assistant){2,}\b', sanitized_content):
                sanitized_content = _re.sub(r'^(?:system|user|assistant){2,}\b', '', sanitized_content, count=1).lstrip()

            last_idx = 0
            for match in pattern.finditer(sanitized_content):
                start, end = match.span()
                # Append text before the match
                hlcontent.append(sanitized_content[last_idx:start])

                full_match_text = match.group(1)
                tag_name = match.group(2)

                # Regex to parse the tag and content
                tag_pattern = _re.compile(r"<(?P<tag_name>\w+)(?P<attributes>[^>]*)>(?P<inner_content>.*?)</\1>", _re.DOTALL)
                tag_match = tag_pattern.match(full_match_text)

                if tag_match:
                    attributes_str = tag_match.group('attributes')
                    inner_content = tag_match.group('inner_content')

                    style = "yellow"
                    if tag_name in ["compress", "decompress", "think", "json", "critique"]:
                        style = "blue"

                    # Reconstruct and style
                    hlcontent.append(f"<{tag_name}{attributes_str}>", style=f"bold {style}")
                    hlcontent.append(inner_content, style=style)
                    hlcontent.append(f"</{tag_name}>", style=f"bold {style}")
                else:
                    # Fallback for non-matching (should not happen with the outer regex)
                    hlcontent.append(full_match_text)

                last_idx = end

            # Append any remaining text
            hlcontent.append(sanitized_content[last_idx:])

            # Render role header
            # ----------------------------------------
            if imsg > 0:
                ret.append("\n", style="white")
                ret.append("\n", style="white")

            # single_line = "\n" not in str(hlcontent)
            # if single_line:
            #     ret.append(f"--- {role} ---", style=color)
            # else:
            ret.append(f"--- {role} ---", style=color)
            ret.append("\n")

            ret.append(hlcontent)

        return ret

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
                tmp_ctx.add_frag(ego="system", content="[empty context]", type=FragType.FROZEN)
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
