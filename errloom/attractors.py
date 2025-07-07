import logging
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.panel import Panel

from errloom import Attractor
from errloom.holophore import Holophore
from errloom.holoware import ClassSpan, TextSpan
from errloom.utils.logging_utils import ellipsis

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
console = Console()


class BingoAttractor(Attractor):
    def __init__(self, holophore: Holophore, span: ClassSpan, dry_run: bool = False):
        super().__init__()
        self.holophore = holophore
        self.span = span
        self.dry_run = dry_run
        self.heuristics: list[str] = []
        self.seed_prompt: Optional[str] = None

    @classmethod
    def __holo_init__(cls, holophore: Holophore, span:ClassSpan, *kargs, **kwargs):
        """
        Called at the beginning of the holoware execution.
        This will decompose the goal into heuristics.
        """
        inst = Attractor.__holo_init__(cls, holophore, span, *kargs, **kwargs)

        goal_prompt = ""
        if span.body:
            for s in span.body.spans:
                if isinstance(s, TextSpan):
                    goal_prompt = s.text
                    break

        if holophore.dry: # TODO we need this from the loom
            console.print(Panel(f"[bold]BingoAttractor Goal:[/] {goal_prompt}", title="BingoAttractor (Dry Run)", expand=False))
            return

        # In a real run, this would call an LLM to decompose the goal.
        # For now, we'll just log that it's happening.
        # logger.debug(f"Decomposing BingoAttractor goal: {ellipsis(goal_prompt)}")
        inst.heuristics = ["use_abbreviations", "mix_languages", "utilize_unicode"]
        inst.seed_prompt = "compress"
        logger.debug("TODO actually call the LLM to decompose the prompt and implement bingo") # TODO call the llm and decompose

    def __holo__(self, **kwargs):
        """
        The main holofunc for the BingoAttractor.
        In a dry run, it does nothing as the goal is printed in __holo_start__.
        In a real run, this is where the reward logic would be applied.
        :param **kwargs:
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