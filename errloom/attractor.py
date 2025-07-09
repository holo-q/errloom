import asyncio
import concurrent.futures
import inspect
import picologging as logging
from asyncio import Semaphore
from typing import Callable, List

from errloom.holophore import Holophore
from errloom.holoware import ClassSpan
from errloom.parsing.parser import Parser
from errloom.tapestry import Rollout, Tapestry

FnRule = Callable[..., float]

# noinspection PyDefaultArgument
class Attractor:
    """
    Attractor class for attraction rules.
    An attractor applies gravitation rules to steer the spoolings
    of the loom and thread sacred geomestries into the weights.

    Each attraction rule takes:
    - prompt: MessageList | str
    - completion: MessageList | str
    - answer: Any (metadata for scoring)
    - task (optional): str (type of task)
    - **kwargs: additional kwargs

    Returns:
    - float | List[float] | Dict[str, float]
    """

    def __init__(self,
                 funcs: List[FnRule] = [],
                 weights: List[float] = [],
                 parser: Parser = Parser()):
        self.logger = logging.getLogger(f"errloom.attractors.{self.__class__.__name__}")
        self.parser = parser
        self._rule_funcs = funcs
        self._rule_weights = weights
        if not self._rule_weights:
            self._rule_weights = [1.0] * len(self._rule_funcs)

    @property
    def rule_names(self) -> List[str]:
        return [func.__name__ for func in self._rule_funcs]

    @property
    def rule_weights(self) -> List[float]:
        return self._rule_weights  # type: ignore

    @property
    def rule_funcs(self) -> List[FnRule]:
        return self._rule_funcs  # type: ignore

    def add_rule(self, func: FnRule, weight: float = 1.0):
        self._rule_funcs.append(func)
        self._rule_weights.append(weight)

    def __holo_init__(self, holophore:Holophore, span:ClassSpan)->'Attractor':
        # TODO you know the drill
        pass

    def __holo__(self, holophore:Holophore, span:ClassSpan):
        # The span contains var_args and var_kwargs that can be used to configure the attractor
        # For now, we create a default instance, but this could be enhanced to use span.var_args/var_kwargs
        # attractor = cls(*span.var_args, **span.var_kwargs)
        self.feel(holophore.rollout)

    def _evaluate_rule(self, func: FnRule, roll: Rollout) -> float:
        """
        Invoke `func` with only the required arguments.

        Example:
        ```
        def func(completion, answer, **kwargs):
            ...
        ``
        """
        assert len(roll.samples) >= 1

        common = dict(
            rollout=roll,
            completion=roll.samples[0],
            prompt=roll.row.get('prompt', None),
            answer=roll.row.get('answer', None),
            task=roll.row.get('task', None),
        )
        merged = {**common, **roll.extra} # TODO what is the meaning of this:   , **kwargs}    ?

        sig = inspect.signature(func)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            try:
                return func(**merged)
            except Exception as e:
                self.logger.error(f"Error calling attraction function {func.__name__}: {e}")
                return 0.0
        else:
            allowed = {k: v for k, v in merged.items() if k in sig.parameters}
            try:
                return func(**allowed)
            except Exception as e:
                self.logger.error(f"Error calling attraction function {func.__name__}: {e}")
                return 0.0


    def feel(self, rollout: Rollout) -> dict[str, float] | float:
        """
        Evaluate all attraction functions asynchronously for a single rollout.
        :param rollout:
        """
        # Use concurrent.futures directly to avoid nested event loop issues
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._evaluate_rule, func, rollout)
                for func in self._rule_funcs
            ]
            gravities = [future.result() for future in futures]
        
        return {func.__name__: gravity for func, gravity in zip(self._rule_funcs, gravities)}
        # rollout.reward += sum([gravity * weight for gravity, weight in zip(gravity_scores, self.get_rule_weights())])

    def feels(self, tapestry: Tapestry):
        """
        Compute attraction gravities for a group of rollouts.

        Default behavior:
        - evaluate each rollout asynchronously
        - return list of dictionaries of attraction rule names and their gravities

        Potential overrides:
        - inter-group comparisons (voting, ranking, Elo, etc.)
        - gravities computed using global state stored in Attractor class
        :param tapestry:
        """
        from tqdm import tqdm
        
        # Use concurrent.futures directly to avoid nested event loop issues
        with concurrent.futures.ThreadPoolExecutor(max_workers=tapestry.max_concurrent) as executor:
            futures = [
                executor.submit(self.feel, rollout)
                for rollout in tapestry.rollouts
            ]
            
            # Use tqdm to show progress
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(tapestry.rollouts), 
                              desc=f"Evaluating {len(tapestry.rollouts)} rollouts"):
                future.result()  # Wait for completion

