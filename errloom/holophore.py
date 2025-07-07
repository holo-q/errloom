from typing import Any, Optional
import typing

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from errloom.utils.openai_chat import extract_fence
from errloom.utils.logging_utils import cl
from errloom.rollout import Context, Rollout

if typing.TYPE_CHECKING:
    from errloom.loom import Loom

class Holophore:
    """
    Binding delegator around Loom and Rollout as a superset execution context for an holoware.
    All field access and modifications are delegated to the original loom and rollout object.
    """

    def __init__(self, loom: 'Loom', rollout: Rollout, env: Optional[dict[str, Any]] = None):
        # Store reference to original loom and rollout for delegation
        self._loom = loom
        self._rollout = rollout
        self.env = env or {}

    def __getattr__(self, name):
        # Delegate attribute access to the original rollout, then loom
        if hasattr(self._rollout, name):
            return getattr(self._rollout, name)
        if hasattr(self._loom, name):
            return getattr(self._loom, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Handle our own attributes
        if name.startswith('_') or name in ['env']:
            super().__setattr__(name, value)
        # Delegate rollout attribute modifications to original rollout
        elif hasattr(self._rollout, name) and not hasattr(self._loom, name):
            setattr(self._rollout, name, value)
        else:
            super().__setattr__(name, value)

    @property
    def contexts(self) -> list[Context]:
        return self._rollout.contexts

    @property
    def loom(self) -> 'Loom':
        """Returns the original loom object."""
        return self._loom

    @property
    def rollout(self) -> Rollout:
        """For backward compatibility - returns the original rollout object."""
        return self._rollout

    def extract_fence(self, tag, role="assistant") -> Optional[str]:
        for c in self.contexts:
            ret = extract_fence(c.messages, tag, role)
            if ret:
                return ret
        return None

    def to_rich(self):
        renderables = []
        for i, ctx in enumerate(self.contexts):
            if i > 0:
                renderables.append(Rule(style="yellow"))
            renderables.append(ctx.text_rich)
        return Group(*renderables)

    def log(self, fidelity_score: float, reward: float) -> None:
        cl.print(Panel(
            self.to_rich(),
            title="[bold yellow]Dry Run: Full Conversation Flow[/]",
            border_style="yellow",
            box=box.ROUNDED,
            title_align="left"
        ))

        # Create a table for beautiful logging
        table = Table(box=box.MINIMAL, show_header=False, expand=True)
        table.add_column(style="bold magenta")
        table.add_column(style="white")

        table.add_row("Fidelity:", f"{fidelity_score:.2f}")
        table.add_row("Reward:", f"{reward:.2f}")
        # table.add_row("Evaluation:", evaluation)

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_row(Panel(
            table,
            title=f"[bold green]Compression Sample[/]",
            border_style="green",
            title_align="left"
        ))
        cl.print(grid)

    def __repr__(self) -> str:
        return f"Holophore(loom={self._loom!r}, rollout={self._rollout!r}, contexts={len(self.contexts)} contexts)"
