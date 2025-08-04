# TODO rename to holoware_loader

import logging
import os
from typing import List, Optional

from errloom.holoware.holoware import Holoware
from errloom.holoware import holoware_parser

logger = logging.getLogger(__name__)
_default_loader = None

class HolowareLoader:
    """
    Utility class for loading, parsing, and formatting prompt templates.
    """

    def __init__(self, search_paths: List[str] = ["prompts", "hol"]):
        self.search_paths = search_paths
        self._cache = {}

    def find_holoware_path(self, filename: str) -> Optional[str]:
        """
        Find the absolute path for a given holoware filename.
        Resolution order:
        1. Absolute path
        2. Relative path from current working directory
        3. Search in predefined library folders
        """
        # 1. Absolute path
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename

        # 2. Relative path from CWD
        if os.path.sep in filename and os.path.exists(filename):
            return filename

        # 3. Search in library folders
        for search_path in self.search_paths:
            full_path = os.path.join(search_path, filename)
            if os.path.exists(full_path):
                return full_path

        return None

    def load_holoware(self, filename: str) -> Holoware:
        """
        Load a prompt from file and parse it if it uses the DSL.
        """

        if filename in self._cache:
            return self._cache[filename]

        prompt_path = self.find_holoware_path(filename)
        if not prompt_path:
            logger.error(f"Holoware file not found: {filename} (searched in {self.search_paths})")
            raise FileNotFoundError(f"[Errno 2] No such file or directory: '{filename}'")

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                text = f.read()

            text = holoware_parser.filter_comments(text)
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
        all_prompts = []
        for search_dir in self.search_paths:
            try:
                all_prompts.extend([f for f in os.listdir(search_dir) if f.endswith('.hol')])
            except OSError:
                # Directory doesn't exist or can't be accessed - this is expected for optional directories
                pass
        return list(set(all_prompts))

def get_default_loader(search_paths: List[str] = ["prompts", "hol"]) -> HolowareLoader:
    """Get or create the default prompt library instance."""
    global _default_loader
    if _default_loader is None or _default_loader.search_paths != search_paths:
        _default_loader = HolowareLoader(search_paths)
    return _default_loader

def load_holoware(filename: str, search_paths: List[str] = ["prompts", "hol"]) -> Holoware:
    """Convenience function to load a prompt using the default library."""
    return get_default_loader(search_paths).load_holoware(filename)
