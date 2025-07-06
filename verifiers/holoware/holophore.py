from typing import Any, Optional

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from verifiers.holoware.openai_chat import extract_fence
from verifiers.log import cl
from verifiers.states import Context, Rollout


class Holophore(Rollout):
    """
    Extension wrapper around Rollout that provides additional holoware functionality.
    All rollout field access and modifications are delegated to the original rollout object.
    """

    def __init__(self, rollout: Rollout, contexts: Optional[list[Context]] = None, env: Optional[dict[str, Any]] = None):
        # Store reference to original rollout for delegation
        self._rollout = rollout
        self.env = env or {}
        # Initialize contexts if provided, otherwise use rollout's contexts
        if contexts is not None:
            self._contexts = contexts
        else:
            self._contexts = rollout.contexts if hasattr(rollout, 'contexts') else []

    def __getattr__(self, name):
        # Delegate all attribute access to the original rollout
        return getattr(self._rollout, name)

    def __setattr__(self, name, value):
        # Handle our own attributes
        if name.startswith('_') or name in ['contexts']:
            super().__setattr__(name, value)
        else:
            # Delegate rollout attribute modifications to original rollout
            setattr(self._rollout, name, value)

    @property
    def contexts(self) -> list[Context]:
        return self._contexts

    @contexts.setter
    def contexts(self, value: list[Context]):
        self._contexts = value

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
        return f"Holophore(rollout={self._rollout!r}, contexts={len(self.contexts)} contexts)"
