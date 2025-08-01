"""
File system watcher for monitoring file changes and executing commands.

This module provides a Watcher class that can monitor specified files and directories
for changes, then execute registered commands when modifications are detected.
It supports both one-time monitoring and continuous background monitoring via threads.

Example usage:
    watcher = Watcher(['/path/to/file.py'], [lambda f: print(f"File {f} changed")])
    watcher.run_monitor()  # Start continuous monitoring
"""

import datetime
import logging
import os
import threading
import time
from typing import Callable, List


log = logging.getLogger(__name__)


class FileWatcher(object):
    """
    A file system watcher that monitors files for changes and executes commands.

    The Watcher can monitor individual files or entire directories recursively.
    When changes are detected, it executes registered commands which can be either
    shell commands (strings) or Python callables. Supports both synchronous and
    threaded continuous monitoring modes.

    Attributes:
        files: List of absolute file paths being monitored
        cmds: List of commands to execute on file changes
        num_runs: Counter for total command executions
        mtimes: Dictionary mapping file paths to their last modification times
        verbose: Whether to log detailed information about file changes
    """

    def __init__(self, files: List[str] = [], cmds: List[Callable] = [], verbose=False):
        """
        Initialize a new Watcher instance.

        Args:
            files: List of file or directory paths to monitor
            cmds: List of commands to execute when files change. Can be strings
                  (shell commands) or callables (Python functions)
            verbose: Enable verbose logging of file changes and command execution
        """
        self.files = []
        self.cmds = []
        self.num_runs = 0
        self.mtimes = {}
        self._monitor_continously = False
        self._monitor_thread = None
        self.verbose = verbose

        if files:
            self.add_files(*files)
        if cmds:
            self.add_cmds(*cmds)

    def monitor(self):
        """
        Start continuous monitoring in a background thread.

        Creates a daemon thread that continuously monitors files for changes.
        Only one monitoring thread is allowed at a time; any existing thread
        will be stopped before starting a new one.
        """
        # We only want one thread, dear god
        self.stop_monitor()

        self._monitor_continously = True
        self._monitor_thread = threading.Thread(target=self.monitor_till_stopped)
        self._monitor_thread.start()

    def run_monitor(self):
        """
        Start continuous monitoring in the main thread with Ctrl-C handling.

        This method blocks the main thread while monitoring files continuously.
        It properly handles KeyboardInterrupt (Ctrl-C) to allow graceful shutdown.
        Use this method when running from __main__ or when you want the main
        thread to be blocked during monitoring.
        """
        self.monitor()
        try:
            while self._monitor_continously:
                time.sleep(0.02)
        except KeyboardInterrupt:
            self.stop_monitor()

    def stop_monitor(self):
        """
        Stop any active monitoring thread.

        Sets the monitoring flag to False and waits for the monitoring thread
        to complete. This method is safe to call even if no thread is running.
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_continously = False
            self._monitor_thread.join(0.05)

    def monitor_till_stopped(self):
        """
        Internal method for continuous monitoring loop.

        This method runs in the monitoring thread and continuously checks
        for file changes until stop_monitor() is called. It sleeps for 1
        second between checks to avoid excessive CPU usage.
        """
        while self._monitor_continously:
            self.monitor_once()
            time.sleep(1)

    def monitor_once(self, execute=True):
        """
        Perform a single check for file changes.

        Args:
            execute: Whether to actually execute commands when changes are detected.
                    Set to False for dry-run mode where only file states are updated.

        Returns:
            None. Side effects include updating mtimes and potentially executing commands.
        """
        for f in self.files:
            try:
                mtime = os.stat(f).st_mtime
            except OSError:
                # The file might be right in the middle of being written so sleep
                time.sleep(1)
                mtime = os.stat(f).st_mtime

            if f not in self.mtimes.keys():
                self.mtimes[f] = mtime
                continue

            if mtime > self.mtimes[f]:
                if self.verbose:
                    log.info("File changed: %s" % os.path.realpath(f))
                self.mtimes[f] = mtime
                if execute:
                    self.execute(f)

    def execute(self, f):
        """
        Execute all registered commands for a changed file.

        Args:
            f: The file path that triggered the command execution

        Returns:
            The new total count of command executions

        Side effects:
            Increments num_runs counter
            Executes all registered commands (shell commands or callables)
        """
        if self.verbose:
            log.info("Running commands at %s" % (datetime.datetime.now(),))
        for c in self.cmds:
            if isinstance(c, str):
                os.system(c)
            elif callable(c):
                c(f)

        self.num_runs += 1
        return self.num_runs

    def walk_dirs(self, dirnames):
        """
        Recursively walk directories and collect all file paths.

        Args:
            dirnames: List of directory paths to walk

        Returns:
            List of absolute file paths found in all directories and subdirectories
        """
        dir_files = []
        for dirname in dirnames:
            for path, dirs, files in os.walk(dirname):
                files = [os.path.join(path, f) for f in files]
                dir_files.extend(files)
                dir_files.extend(self.walk_dirs(dirs))
        return dir_files

    def add_files(self, *files):
        """
        Add files or directories to the watch list.

        Accepts both individual files and directories. Directories are walked
        recursively to include all contained files. Only existing files are added,
        and duplicates are automatically filtered out.

        Args:
            *files: Variable number of file or directory paths to add

        Side effects:
            Updates self.files with new unique file paths
            Performs an initial dry-run check to establish baseline mtimes
        """
        dirs = [os.path.realpath(f) for f in files if os.path.isdir(f)]
        files = [os.path.realpath(f) for f in files if os.path.isfile(f)]

        dir_files = self.walk_dirs(dirs)
        files.extend(dir_files)

        valid_files = [os.path.realpath(f) for f in files if os.path.exists(f) and os.path.isfile(f)]
        unique_files = [f for f in valid_files if f not in self.files]
        print(f"Adding {len(unique_files)} files to watch")
        self.files = self.files + unique_files
        self.monitor_once(execute=False)

    def add_cmds(self, *cmds):
        """
        Add commands to execute when files change.

        Accepts both shell commands (strings) and Python callables. Callables
        receive the changed file path as their single argument. Duplicates
        are automatically filtered out.

        Args:
            *cmds: Variable number of commands to add

        Side effects:
            Updates self.cmds with new unique commands
        """
        unique_cmds = [c for c in cmds if c not in self.cmds]
        self.cmds = self.cmds + unique_cmds
