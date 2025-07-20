import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Global directory tracker
_dir_registry = {}
_next_dir_id = 0

@dataclass
class ScriptEntry:
    """Represents a Python script file within a session"""
    name: str  # Display name (relative to session root)
    path: Path  # Full path to script
    is_main: bool  # Whether this is the main script.py
    last_modified: float
    last_modified_str: str

class SessionEntry:
    """
    A cached representation of a session directory for display
    in menus and lists.
    """
    def __init__(self, path):
        from errloom.session import Session

        self.path = path
        self.last_modified = os.path.getmtime(path)
        self.last_modified_str = time.ctime(self.last_modified)
        self.session = Session.Unloaded(path)
        self.size = self.session.get_total_megabytes()

        # Cache available scripts
        self.scripts: List[ScriptEntry] = self._find_scripts()

        # Get or assign directory ID and corresponding color
        parent_dir = os.path.dirname(path)
        if parent_dir not in _dir_registry:
            global _next_dir_id
            _dir_registry[parent_dir] = _next_dir_id
            _next_dir_id += 1

        dir_id = _dir_registry[parent_dir]
        self.color = src.color_util.color_list[dir_id]

        # TODO add code which checks the session for every resource that should be transformed and verify whether they are to produce bake status has_signal_baked and has_init_bake


    def load(self):
        from errloom.session import Session
        self.session = Session.Unloaded(self.path)
        self.size = self.session.get_total_megabytes()

    def _find_scripts(self) -> List[ScriptEntry]:
        """Scan session directory for Python scripts"""
        scripts = []
        session_dir = Path(self.path)

        # Check for main script.py
        main_script = session_dir / "script.py"
        if main_script.exists():
            scripts.append(ScriptEntry(
                name="script.py",
                path=main_script,
                is_main=True,
                last_modified=main_script.stat().st_mtime,
                last_modified_str=time.ctime(main_script.stat().st_mtime)
            ))

        # Check .scripts directory
        scripts_dir = session_dir / ".scripts"
        if scripts_dir.exists():
            for script_path in scripts_dir.glob("**/*.py"):
                rel_path = script_path.relative_to(session_dir)
                scripts.append(ScriptEntry(
                    name=str(rel_path),
                    path=script_path,
                    is_main=False,
                    last_modified=script_path.stat().st_mtime,
                    last_modified_str=time.ctime(script_path.stat().st_mtime)
                ))

        return sorted(scripts, key=lambda x: (not x.is_main, x.name))

    @property
    def name(self):
        return self.session.name
