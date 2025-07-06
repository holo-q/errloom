from typing import List

from errloom.parsers import XMLParser
from errloom.attractors import Attractor
from errloom.attractors.attractor import FnRule

think_answer_parser = XMLParser(fields=["think", "answer"])

def answer_equality_rule(completion, answer) -> float:
    """Attraction rule that checks if the final answer matches the expected answer."""
    try:
        from math_verify import parse, verify  # type: ignore
        response = think_answer_parser.parse_answer(completion)
        return 1.0 if verify(parse(answer), parse(response)) else 0.0
    except Exception:
        return 0.0

class MathAttractor(Attractor):
    def __init__(self,
                 funcs: List[FnRule] = [],
                 weights: List[float] = [],
                 parser: XMLParser | None = None):
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.add_rule(answer_equality_rule)
        self.add_rule(self.parser.get_format_attraction_rule(), weight=0.2)
