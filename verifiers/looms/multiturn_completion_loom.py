from abc import abstractmethod
from copy import deepcopy
from typing import List, Dict, Any, Tuple

from openai import OpenAI

from verifiers.envs.loom import Loom
from verifiers.states import Rollout


class MultiTurnCompletionLoom(Loom):
    def __init__(self, max_turns: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns

    @abstractmethod
    def is_completed(self,
                     prompt: str,
                     state: Dict[str, Any],
                     **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def env_response(self,
                     prompt: str,
                     state: Dict[str, Any],
                     **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response from the environment (message, state).
        """
        pass

    def run(self, state, sampling_args: Dict[str, Any] = {}, **kwargs: Dict[str, Any]) -> Rollout:
        is_completed = False
        state = {'answer': row["answer"]}
        assert isinstance(row["prompt"], str)
        input = deepcopy(row["prompt"])
        completion = ""
        turn = 0
        while not is_completed:
            response = self.sample(
                context=input,
                sampling_args=sampling_args
            )
            input = input + response
            completion += response
            turn += 1
            if self.is_completed(input, state, **kwargs) or turn >= self.max_turns:
                is_completed = True
            else:
                env_msg, state = self.env_response(input, state, **kwargs)
                input = input + env_msg
                completion += env_msg
        return Rollout(row, samples=completion, answer=row["answer"])