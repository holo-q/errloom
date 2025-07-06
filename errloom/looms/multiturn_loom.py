from abc import abstractmethod
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Union

from openai import OpenAI

from errloom.looms.loom import Loom
from errloom.states import Rollout


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

    def run(self, state, sampling_args: Dict[str, Any] = {}, **kwargs: Dict[str, Any]) -> Rollout:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        :param state:
        """
        is_completed = False
        state = {'answer': row["answer"]}
        assert isinstance(row["prompt"], list)
        messages = deepcopy(row["prompt"])
        completion = []
        turn = 0
        while not is_completed:
            if self.is_completed(messages, state, **kwargs):
                is_completed = True
                break
            response = self.sample(
                context=messages,
                sampling_args=sampling_args,
            )
            has_error = response.startswith("[ERROR]")
            messages.append({"role": "assistant", "content": response})
            completion.append({"role": "assistant", "content": response})
            turn += 1
            if self.is_completed(messages, state, **kwargs) or turn >= self.max_turns or has_error:
                is_completed = True
            else:
                env_msg, state = self.loom_response(messages, state, **kwargs)
                messages.append(env_msg)
                completion.append(env_msg)
        return Rollout(samples=completion, answer=row["answer"])