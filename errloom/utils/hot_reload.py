import importlib
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Callable

from errloom import paths
from src.lib import loglib

log = logging.getLogger('hotreload')

class HotReloader:
    """Manages hot-reloading of Python modules"""

    def __init__(self,
                 watch_paths: List[Path],
                 allowed_prefixes: List[str],
                 check_interval: float = 1.0):
        """
        Args:
            watch_paths: List of paths to watch for changes
            allowed_prefixes: List of module path prefixes that are allowed to be reloaded
                            (e.g. ['src.party.', 'tests.'])
            check_interval: Minimum time between checks in seconds
        """
        self.watch_paths = watch_paths
        self.allowed_prefixes = allowed_prefixes
        self.check_interval = check_interval
        self._script_time_cache: Dict[str, float] = {}
        self.on_module_reload: List[Callable[[str], None]] = []
        self.on_modules_reloading: List[Callable[[str], None]] = []
        self.on_modules_reloaded: List[Callable[[str], None]] = []
        self._last_check_time = 0
        self._collect_changes() # on the first call all the files detect as added

    def apply_hot_reload(self) -> bool:
        """
        Check for changes and reload modified modules if enough time has passed
        Returns:
            Whether any modules were reloaded
        """
        current_time = time.time()
        if current_time - self._last_check_time < self.check_interval:
            return False

        self._last_check_time = current_time
        return self._check_and_reload()

    def _check_and_reload(self) -> bool:
        """
        Internal method to check for changes and reload modules
        Returns:
            Whether any modules were reloaded
        """
        changed_files, added_files = self._collect_changes()
        if not changed_files and not added_files:
            return False

        reloaded = False
        for file in changed_files + added_files:
            # Convert file path to module path
            rel_path = None
            for watch_path in self.watch_paths:
                try:
                    rel_path = file.relative_to(watch_path)
                    break
                except ValueError:
                    continue

            if rel_path is None:
                continue

            module_path = paths.filepath_to_modpath(file)

            # Add parent path prefix
            for prefix in self.allowed_prefixes:
                if module_path.startswith(prefix.replace('.', '/')):
                    module_path = prefix + module_path[len(prefix.replace('.', '/')):]
                    break

            if self.reload_module(module_path):
                reloaded = True
                log.info(f"Reloaded module: {module_path}")

            for callback in self.on_modules_reloading: callback(module_path)
            for callback in self.on_modules_reloaded: callback(module_path)

        return reloaded

    def _collect_changes(self) -> Tuple[List[Path], List[Path]]:
        """
        Detect changes in watched paths
        Returns:
            Tuple of (changed_files, added_files)
        """
        changed_files = []
        added_files = []

        for watch_path in self.watch_paths:
            if not watch_path.exists():
                continue

            for file in watch_path.rglob('*.py'):
                if file.name.startswith('__'):  # Skip __init__.py etc
                    continue

                key = str(file)
                mtime = file.stat().st_mtime

                if key not in self._script_time_cache:
                    self._script_time_cache[key] = mtime
                    added_files.append(file)
                elif mtime > self._script_time_cache[key]:
                    self._script_time_cache[key] = mtime
                    changed_files.append(file)

        return changed_files, added_files

    def reload_module(self, module_path: str) -> bool:
        """
        Reload a specific module
        Returns:
            Success status
        """
        try:
            # Check if module is allowed to be reloaded
            if not any(module_path.startswith(prefix) for prefix in self.allowed_prefixes):
                return False

            # Import module if not already imported
            if module_path not in sys.modules:
                importlib.import_module(module_path)
                return True

            # Reload existing module
            module = sys.modules[module_path]
            importlib.reload(module)

            # Notify listeners
            for callback in self.on_module_reload: callback(module_path)

            return True

        except Exception as e:
            log.info(f"Error reloading module {module_path}: {str(e)}")
            return False

    def force_reload(self, module_path: str) -> bool:
        """
        Force reload a specific module regardless of file changes
        Returns:
            Success status
        """
        return self.reload_module(module_path)