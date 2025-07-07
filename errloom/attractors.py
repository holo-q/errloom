import logging
from typing import Optional, TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel

from errloom import Attractor
from errloom.holophore import Holophore
from errloom.holoware import ClassSpan, TextSpan

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
console = Console()

class BingoAttractor(Attractor):
    def __init__(self, holophore: Holophore, span: ClassSpan):
        super().__init__()
        self.holophore = holophore
        self.span = span
        self.heuristics: list[str] = []
        self.seed: Optional[str] = None

    @classmethod
    def __holo_init__(cls, holophore: Holophore, span:ClassSpan):
        """
        Called at the beginning of the holoware execution.
        This will decompose the goal into heuristics.
        """
        inst = cls(holophore, span)
        inst = inst.__holo_init__(holophore, span)
        logger.debug(f"BingoAttractor __holo_init__: {inst}")

        # noinspection PyUnresolvedReferences
        goal = span.body.first_span_by_type(TextSpan).text

        inst.heuristics = ["use_abbreviations", "mix_languages", "utilize_unicode"]
        inst.seed = "compress"
        inst.goal = goal

        if holophore.dry:
            logger.debug(Panel(f"[bold]BingoAttractor Goal:[/] {goal}", title="BingoAttractor (Dry Run)", expand=False))
            return inst

        # TODO call an LLM to decompose the goal.
        logger.debug("TODO actually call the LLM to decompose the prompt and implement bingo") # TODO call the llm and decompose
        return inst

    @classmethod
    def __holo__(cls, holophore, span, instance):
        """
        The main holofunc for the BingoAttractor.
        In a dry run, it does nothing as the goal is printed in __holo_start__.
        In a real run, this is where the reward logic would be applied.
        :param instance:
        :param **kwargs:
        """
        if holophore.rollout.dry:
            return None

        # The core logic for influencing the generation will go here.
        # For now, it just returns the body of the attractor.
        if span.body:
            return span.body(holophore.state, lambda: "MOCK", holophore.env)

        return None

    def __holo_end__(self):
        """
        Called at the end of the holoware execution.
        """
        logger.debug(f"BingoAttractor finished for span {self.span.uuid}")