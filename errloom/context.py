from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional

import errloom.lib.formatting
from errloom.aliases import APIChat
from errloom.lib import formatting, log

logger = log.getLogger(__name__)
logger_convert = log.getLogger(__name__ + ".convert").disable()

class FragType(Enum):
    """Type of text fragment for training purposes."""
    FROZEN = "frozen"  # Input text, typically masked
    REINFORCE = "reinforce"  # Text to reinforce (unmasked)

@dataclass
class Frag:
    """
    A fragment of text with training metadata.
    """
    text: str
    ego: Optional[str]
    type: FragType

    def __str__(self):
        return f"Frag({self.ego}->{errloom.lib.formatting.ellipse(self.text, 50)})"

    def __repr__(self):
        return f"Frag({self.ego}->{errloom.lib.formatting.ellipse(self.text, 50)})"

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

    def add_frag(self, ego: Optional[str], text: str, type: FragType, prints: bool = True) -> Frag:
        """Add a text fragment with training metadata."""
        frag = Frag(text=text, type=type, ego=ego)
        self.fragments.append(frag)
        if prints:
            logger.debug(f"add_fragment({type._name_}) :: {ego} -> {formatting.frag(text)}")
        return frag

    def add_frozen(self, role: Optional[str], text: str) -> Frag:
        """Add text to mask (ignored in training)."""
        ret = self.add_frag(role, text, FragType.FROZEN, prints=False)
        logger.debug(f"add_frozen :: [black on white]{role} -> {formatting.frag(text)}[/]")
        return ret

    def add_reinforced(self, role: Optional[str], text: str) -> Frag:
        """Add text to reinforce (unmasked in training)."""
        ret = self.add_frag(role, text, FragType.REINFORCE, prints=False)
        logger.debug(f"add_reinforced: [black on white]{role} -> {formatting.frag(text)}[/]")
        return ret

    def to_api_messages(self, render_dry: bool = False) -> APIChat:
        """
        Convert context fragments to OpenAI API chat messages by aggregating
        consecutive fragments with the same normalized role.
        Includes display-only sanitation to avoid role-label concatenation leaking into text.

        When render_dry is True, do not drop empty/whitespace-only messages so the dry-run
        view can show scaffolded messages that are masked during training.
        """
        logger = logger_convert

        def _normalize_role(raw: Optional[str], is_first: bool) -> str:
            # TODO print warnings
            if raw in ("system", "user", "assistant"):
                return raw
            if raw is None and is_first:
                return "system"
            return "user"

        messages: APIChat = []
        texts: list[str] = []
        current_role: Optional[str] = None

        logger.debug(f"to_api_messages: Processing {len(self.fragments)} fragments")

        # TODO(agent): there may be some code here that we can move to FragList

        for i, frag in enumerate(self.fragments):
            normalized_role = _normalize_role(frag.ego, is_first=(i == 0))
            ellipsed_content = errloom.lib.formatting.ellipse(frag.text.replace('\n', '\\n'), 45).strip()
            logger.debug(f"- Fragment {i}: [dim]{frag.ego}[/]->{normalized_role} :: [dim]{ellipsed_content}[/]")

            if current_role is None:
                # First fragment
                current_role = normalized_role
                texts = [frag.text]
            elif normalized_role == current_role:
                # Same role, accumulate content
                texts.append(frag.text)
            else:
                # Role changed, finalize previous message
                s = "".join(texts)
                if s or render_dry:
                    messages.append({"role": current_role, "content": s.strip()})
                    # logger.debug(f"Role changed from {current_role} to {normalized_role}, added message with {len(s)} chars")
                current_role = normalized_role
                texts = [frag.text]

        # flush tail
        if current_role is not None:
            s = "".join(texts)
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
            content_text = msg.get("content", "")

            blocks.append(f"<|im_start|>{role}\n{content_text}\n<|im_end|>")

        # TODO(agent): there may be some code here that we can move to FragList

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
                last_text = self.fragments[last_idx].text or ""
                if last_text == "":
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
            context.add_frag(ego=role, text=content, type=ftype)
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
