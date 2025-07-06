from typing import Any, Tuple, List

from datasets import Dataset
from openai import OpenAI
import reasoning_gym as rg
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset

from errloom.parsers import XMLParser
from errloom.attractors import Attractor
from errloom.looms.qa_loom import QuestionAnswerLoom

class ReasoningGymLoom(QuestionAnswerLoom):
    def __init__(self,
                 gym: str | List[str | dict],
                 num_samples: int = 1000,
                 num_eval_samples: int = 100,
                 seed: int = 0,
                 **kwargs: Any):
        self.gym = gym
        self.num_samples = num_samples
        self.num_eval_samples = num_eval_samples
        self.seed = seed
        total_samples = num_samples + num_eval_samples
        self.rg_dataset = self.build_rg_dataset(gym, num_samples=total_samples, seed=seed)
        dataset, eval_dataset = self.rg_to_hf(self.rg_dataset)
        parser = XMLParser(fields=['think', 'answer'])
        attractor = Attractor(parser=parser)
        def check_answer_attraction_rule(completion, answer, **kwargs) -> float:
            entry = self.rg_dataset[answer]
            response = str(parser.parse_answer(completion)).strip()
            reward = self.rg_dataset.score_answer(answer=response, entry=entry)
            return reward
        attractor.add_rule(check_answer_attraction_rule)
        attractor.add_rule(parser.get_format_attraction_rule(), weight=0.2)
        system_prompt = rg.utils.SYSTEM_PROMPTS["DeepSeekZero"] # type: ignore
        super().__init__(
            roll_dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=parser,
            attractor=attractor,
            message_type='chat',
            **kwargs
        )
        self.parser = parser
        self.attractor = attractor

    def build_rg_dataset(self, gym: str | List[str | dict], num_samples: int = 1000, seed: int = 0) -> ProceduralDataset:
        if isinstance(gym, str):
            return rg.create_dataset(gym, size=num_samples, seed=seed)
        if not isinstance(gym, list):
            raise ValueError("'gym' must be str or list")
        dataset_specs = []
        for dataset_config in gym:
            if isinstance(dataset_config, str):
                dataset_specs.append(DatasetSpec(name=dataset_config, weight=1.0, config={}))
            elif isinstance(dataset_config, dict):
                dataset_specs.append(DatasetSpec(**dataset_config))
            else:
                raise ValueError(f"Invalid dataset config: {dataset_config}")
        return rg.create_dataset("composite", datasets=dataset_specs, size=num_samples, seed=seed)

    def rg_to_hf(self, rg_dataset: ProceduralDataset) -> Tuple[Dataset, Dataset]:
        dataset_rows = []
        eval_rows = []
        for i, x in enumerate(rg_dataset):
            row = {
                'question': x['question'],
                'answer': i,
                'task': x['metadata']['source_dataset'],
            }
            if i < self.num_samples:
                dataset_rows.append(row)
            else:
                eval_rows.append(row)
        dataset = Dataset.from_list(dataset_rows)
        eval_dataset = Dataset.from_list(eval_rows)
        return dataset, eval_dataset