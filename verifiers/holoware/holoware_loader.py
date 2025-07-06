import logging
import os
from typing import List, Union

from verifiers.holoware.holoware import Holoware

logger = logging.getLogger(__name__)
_default_loader = None

class HolowareLoader:
    """
    Utility class for loading, parsing, and formatting prompt templates.
    """

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self._cache = {}

    def load_holoware(self, filename: str) -> Holoware:
        """
        Load a prompt from file and parse it if it uses the DSL.
        """
        import verifiers.holoware.holoware_parse

        if filename in self._cache:
            return self._cache[filename]

        prompt_path = os.path.join(self.prompts_dir, filename)
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # First, filter out comment lines
            text = holoware_parse.filter_comments(text)  # TODO do this as part of _parse_prompt
            tpl = Holoware.parse(text)

            self._cache[filename] = tpl
            return tpl

        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt from {prompt_path}: {e}")
            raise

    def clear_cache(self):
        self._cache.clear()

    def list_prompts(self) -> List[str]:
        try:
            return [f for f in os.listdir(self.prompts_dir) if f.endswith('.txt')]
        except OSError:
            logger.warning(f"Could not list prompts directory: {self.prompts_dir}")
            return []

def get_default_loader(prompts_dir: str = "prompts") -> HolowareLoader:
    """Get or create the default prompt library instance."""
    global _default_loader
    if _default_loader is None or _default_loader.prompts_dir != prompts_dir:
        _default_loader = HolowareLoader(prompts_dir)
    return _default_loader

def load_holoware(filename: str, prompts_dir: str = "prompts") -> Union[Holoware, str]:
    """Convenience function to load a prompt using the default library."""
    return get_default_loader(prompts_dir).load_holoware(filename)
