from typing import List, Dict, Any, Tuple

from datasets import Dataset
from openai import OpenAI

from errloom.utils.openai_chat import MessageList
from errloom import FnRule
from errloom.looms.multiturn_loom import MultiTurnLoom
from errloom.hol.system_prompts import SIMPLE_PROMPT
from errloom.attractors import MathAttractor

class DoubleCheckLoom(MultiTurnLoom):
    def __init__(self,
                 client: OpenAI | None = None,
                 model: str | None = None,
                 data_train: Dataset | None = None,
                 data_bench: Dataset | None = None,
                 system_prompt: str = SIMPLE_PROMPT,
                 few_shot: MessageList = [],
                 **kwargs):
        super().__init__(
            client=client,
            model=model,
            dataset=data_train,
            eval_dataset=data_bench,
            system_prompt=system_prompt,
            few_shot=few_shot,
            **kwargs
        )
        self.attractor = MathAttractor()

    def get_attraction_rules(self, **kwargs: Any) -> List[FnRule]:
        return self.attractor.get_rule_funcs()

    def get_rule_weights(self, **kwargs: Any) -> List[float]:
        return self.attractor.get_rule_weights()

    def is_completed(self,
                     messages: MessageList,
                     state: Dict[str, Any],
                     **kwargs: Any) -> bool:
        return len(messages) > 1 and messages[-2]['content'] == 'Are you sure?'

    def loom_response(self,
                      messages: MessageList,
                      state: Dict[str, Any],
                      **kwargs: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
        return {'role': 'user', 'content': 'Are you sure?'}, state