import asyncio
import concurrent.futures
import inspect
import logging
from asyncio import Semaphore
from typing import Callable, List

from errloom.holophore import Holophore
from errloom.holoware import ObjSpan
from errloom.parsers.parser import Parser
from errloom.rollout import Rollout, Rollouts

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
        self.rule_funcs = funcs
        self.rule_weights = weights
        if not self.rule_weights:
            self.rule_weights = [1.0] * len(self.rule_funcs)

    @property
    def rule_names(self) -> List[str]:
        return [func.__name__ for func in self.rule_funcs]

    @property
    def rule_weights(self) -> List[float]:
        return self.rule_weights  # type: ignore

    @property
    def rule_funcs(self) -> List[FnRule]:
        return self.rule_funcs  # type: ignore

    def add_rule(self, func: FnRule, weight: float = 1.0):
        self.rule_funcs.append(func)
        self.rule_weights.append(weight)

    @classmethod
    def __holo__(cls, holophore:Holophore, span:ObjSpan):
        # The span contains var_args and var_kwargs that can be used to configure the attractor
        # For now, we create a default instance, but this could be enhanced to use span.var_args/var_kwargs
        attractor = cls(*span.var_args, **span.var_kwargs)
        attractor.feel(holophore.rollout)

        holophore.contexts[-1].attractors.append(attractor)
        return None

    def _evaluate_rule(self, func: FnRule, roll: Rollout, **kwargs) -> float:
        """
        Invoke `func` with only the required arguments.

        Example:
        ```
        def func(completion, answer, **kwargs):
            ...
        ``
        :param bag:
        """
        assert len(roll.samples) >= 1

        common = dict(
            rollout=roll,
            completion=roll.samples[0],
            prompt=roll.row.get('prompt', None),
            answer=roll.row.get('answer', None),
            task=roll.row.get('task', None),
        )
        merged = {**common, **roll.extra, **kwargs}

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


    async def _evaluate_single(self, semaphore, rollout: Rollout):
        async with semaphore:
            await self.feel(rollout)

    async def _evaluate_all(self, rollouts: Rollouts, max_concurrent: int = 1024):
        from tqdm.asyncio import tqdm
        semaphore = Semaphore(max_concurrent)
        tasks = [
            asyncio.create_task(self._evaluate_single(semaphore, rollout))
            for rollout in rollouts.rollouts
        ]

        for future in tqdm.as_completed(
            tasks,
            total=len(rollouts.rollouts),
            desc=f"Evaluating {len(rollouts.rollouts)} rollouts"
        ):
            await future

    def feels(self, rollouts: Rollouts):
        """
        Compute attraction gravities for a group of rollouts.

        Default behavior:
        - evaluate each rollout asynchronously
        - return list of dictionaries of attraction rule names and their gravities

        Potential overrides:
        - inter-group comparisons (voting, ranking, Elo, etc.)
        - gravities computed using global state stored in Attractor class
        :param rollouts:
        """

        # Set up custom executor for the event loop if needed
        def setup_executor(loop):
            if loop._default_executor is None:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=rollouts.max_concurrent)
                loop.set_default_executor(executor)

        coro = self._evaluate_all(rollouts, max_concurrent=rollouts.max_concurrent)
        try:
            # Create new event loop with custom executor
            loop = asyncio.new_event_loop()
            setup_executor(loop)
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except RuntimeError:
            # Jupyter notebook or existing event loop
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            setup_executor(loop)
            return loop.run_until_complete(coro)

    async def feel(self, rollout: Rollout) -> dict[str, float] | float:
        """
        Evaluate all attraction functions asynchronously for a single rollout.
        :param rollout:
        """
        futures = [
            asyncio.to_thread(self._evaluate_rule, func, rollout)
            for func in self.rule_funcs
        ]
        gravities = await asyncio.gather(*futures)
        return {func.__name__: gravity for func, gravity in zip(self.rule_funcs, gravities)}
        # rollout.reward += sum([gravity * weight for gravity, weight in zip(gravity_scores, self.get_rule_weights())])