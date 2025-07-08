import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

logger = logging.getLogger(__name__)

Message: TypeAlias = Dict[str, Any]
MessageList: TypeAlias = List[Message]
MessageTuple: TypeAlias = Tuple[Message]
ContextType: TypeAlias = Union[str, List[Message]]

# TODO take a ContextType
def extract_fence(context: MessageList, wrapper_tag: Optional[str], role='assistant') -> Optional[str]:
    """Extract content from a dynamic wrapper tag, e.g., <compress>...</compress>"""
    if not wrapper_tag:
        return context.strip() if isinstance(context, str) else context[-1]["content"].strip()

    tag = wrapper_tag.lower()

    for msg in reversed(context):
        if role is not None and msg["role"] != role:
            continue
        content = msg["content"]
        matches = list(re.finditer(fr'<{tag}>\s*(.*?)\s*(?:</{tag}>|$)', content, re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()

    return None

def extract_json(context: ContextType) -> Optional[str]:
    """Extract JSON from anywhere in the response.

    Args:
        context: Either a string or list of messages to extract JSON from

    Returns:
        Extracted JSON string or None if not found
    """
    if isinstance(context, list):
        # If it's a message list, extract content from the last assistant message
        for msg in reversed(context):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                break
        else:
            return None
    else:
        content = context

    # Look for ```json blocks first
    block_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if block_match:
        ret = block_match.group(1).strip()
    else:
        # Look for standalone JSON object
        match = re.search(r'\{[^{}]*(?:\{[^{}]*}[^{}]*)*}', content, re.DOTALL)
        if match:
            ret = match.group(0)
        else:
            logger.warning("No JSON found in evaluation")
            return None

    return ret

def format_conversation(conversation: MessageList):
    text = ""
    for msg in conversation:
        text += msg['content']
    return text
