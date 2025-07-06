import logging
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.panel import Panel

from errloom.utils.logging_utils import ellipsis

if TYPE_CHECKING:
    from errloom.holoware import ClassSpan
    from errloom.holophore import Holophore
else:
    from errloom.holoware import TextSpan

logger = logging.getLogger(__name__)
console = Console()


class BingoAttractor:
    def __init__(self, holophore: "Holophore", span: "ClassSpan", dry_run: bool = False):
        self.holophore = holophore
        self.span = span
        self.dry_run = dry_run
        self.heuristics: list[str] = []
        self.seed_prompt: Optional[str] = None

    def __holo_start__(self):
        """
        Called at the beginning of the holoware execution.
        This will decompose the goal into heuristics.
        """
        goal_prompt = ""
        if self.span.body:
            for s in self.span.body.spans:
                if isinstance(s, TextSpan):
                    goal_prompt = s.text
                    break

        if self.dry_run:
            console.print(Panel(f"[bold]BingoAttractor Goal:[/] {goal_prompt}", title="BingoAttractor (Dry Run)", expand=False))
            return

        # In a real run, this would call an LLM to decompose the goal.
        # For now, we'll just log that it's happening.
        logger.debug(f"Decomposing BingoAttractor goal: {ellipsis(goal_prompt)}")
        self.heuristics = ["use_abbreviations", "mix_languages", "utilize_unicode"]
        self.seed_prompt = "compress"

    def __holo__(self):
        """
        The main holofunc for the BingoAttractor.
        In a dry run, it does nothing as the goal is printed in __holo_start__.
        In a real run, this is where the reward logic would be applied.
        """
        if self.dry_run:
            return None

        # The core logic for influencing the generation will go here.
        # For now, it just returns the body of the attractor.
        if self.span.body:
            return self.span.body(self.holophore.state, lambda: "MOCK", self.holophore.env)

        return None

    def __holo_end__(self):
        """
        Called at the end of the holoware execution.
        """
        logger.debug(f"BingoAttractor finished for span {self.span.uuid}") 