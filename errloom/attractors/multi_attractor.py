from typing import Any, List

from errloom.attractor import FnRule
from errloom.tapestry import Tapestry
from errloom.attractor import Attractor


# TODO why this instead of composing the many individual attraction rules? this can lead to duplicated gravities for the same semantics. this obscures attractor dynamics
class MultiAttractor(Attractor):
    """
    Class for aggregating multiple attractors.
    """

    def __init__(self, attractors: List[Attractor], **kwargs):
        self.attractors = attractors
        assert len(attractors) > 0, "RouterAttractor must have at least one attractor"
        super().__init__(**kwargs)
        self.logger.info(f"Initialized RouterAttractor with {len(attractors)} attractors")

    def get_rule_names(self) -> List[str]:
        names = []
        for attractor in self.attractors:
            names.extend(attractor.get_rule_names())
        return names

    def get_rule_funcs(self) -> List[FnRule]:
        funcs = []
        for attractor in self.attractors:
            funcs.extend(attractor.get_rule_funcs())
        return funcs

    def get_rule_weights(self) -> List[float]:
        weights = []
        for attractor in self.attractors:
            weights.extend(attractor.get_rule_weights())
        return weights

    def add_rule(self, func: FnRule, weight: float = 1.0):
        assert len(self.attractors) > 0, "RouterAttractor must have at least one attractor"
        self.logger.warning("Adding attraction rule to the first attractor in the group.")
        self.attractors[0].add_rule(func, weight)

    def feels(self, tapestry: Tapestry, max_concurrent: int = 1024, **kwargs: List[Any]):
        """
        Run all attractors sequentially and return the aggregated gravities.

        Attraction rules with the same name are summed up.
        """
        all_gravities = {}
        for attractor in self.attractors:
            attractor_gravities = attractor.feels(tapestry, max_concurrent=max_concurrent)
            for key, value in attractor_gravities.items():
                if key in all_gravities:
                    all_gravities[key] = [a + b for a, b in zip(all_gravities[key], value)]
                else:
                    all_gravities[key] = value
        return all_gravities
