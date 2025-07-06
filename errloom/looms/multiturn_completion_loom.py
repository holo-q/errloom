from abc import abstractmethod
from copy import deepcopy
from typing import List, Dict, Any, Tuple

from openai import OpenAI

from errloom.loom import Loom
from errloom.rollout import Rollout


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

    def run(self, rollout: Rollout) -> Rollout:
        is_completed = False
        row = rollout.row
        state = {'answer': row["answer"]}
        assert isinstance(row["prompt"], str)
        input = deepcopy(row["prompt"])
        completion = ""
        turn = 0
        while not is_completed:
            response = self.sample(
                rollout=rollout,
            )
            input = input + response
            completion += response
            turn += 1
            if self.is_completed(input, state) or turn >= self.max_turns:
                is_completed = True
            else:
                env_msg, state = self.env_response(input, state)
                input = input + env_msg
                completion += env_msg
        
        rollout.samples = completion
        rollout.extra = state
        return rollout