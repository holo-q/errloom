from abc import abstractmethod
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Union

from openai import OpenAI

from errloom.loom import Loom
from errloom.tapestry import Rollout


class MultiTurnLoom(Loom):
    def __init__(self, max_turns: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns

    @abstractmethod
    def is_completed(self,
                     messages: List[Dict[str, Any]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def loom_response(self,
                      messages: List[Dict[str, Any]],
                      state: Dict[str, Any],
                      **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate a response from the environment (message, state).
        """
        pass

    def rollout(self, rollout: Rollout) -> Rollout:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        :param rollout:
        """
        is_completed = False
        row = rollout.row
        state = {'answer': row["answer"]}
        assert isinstance(row["prompt"], list)
        messages = deepcopy(row["prompt"])
        completion = []
        turn = 0
        while not is_completed:
            if self.is_completed(messages, state):
                is_completed = True
                break
            response = self.sample(
                rollout=rollout,
            )
            has_error = response.startswith("[ERROR]")
            messages.append({"role": "assistant", "content": response})
            completion.append({"role": "assistant", "content": response})
            turn += 1
            if self.is_completed(messages, state) or turn >= self.max_turns or has_error:
                is_completed = True
            else:
                env_msg, state = self.loom_response(messages, state)
                messages.append(env_msg)
                completion.append(env_msg)

        rollout.samples = completion
        rollout.extra = state
        return rollout