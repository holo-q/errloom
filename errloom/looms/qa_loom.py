from typing import Any, Dict, List, Tuple, Union

from datasets import Dataset

from errloom.loom import Loom
from errloom.rollout import Rollout

class QuestionAnswerLoom(Loom):
    """
    Environment for tasks that can be framed as question-answering.
    It implements a generic `generate` method that runs rollouts and scores them.
    """

    def __init__(self,
                 data_train: Dataset | None = None,
                 data_bench: Dataset | None = None,
                 question_key: str = "question",
                 answer_key: str = "answer",
                 few_shot: list[dict[str, Any]] | None = None,
                 **kwargs: Any):
        super().__init__(data_train=data_train, data_bench=data_bench, **kwargs)

        if few_shot is None:
            few_shot = []

        # TODO didn't we have a neat format_dataset_qa function?
        if self.message_type == 'chat':
            if self.dataset is not None:
                self.dataset = self.format_dataset(self.dataset, self.system_prompt, few_shot, question_key, answer_key)
            if self.eval_dataset is not None:
                self.eval_dataset = self.format_dataset(self.eval_dataset, self.system_prompt, few_shot, question_key, answer_key)
        else:
            if self.system_prompt or few_shot:
                raise ValueError(
                    'The fields "system_prompt" and "few_shot" are not supported for completion tasks.' \
                    'Please use message_type="chat" instead, or pre-format your dataset ' \
                    'to contain "prompt" and "answer" columns.'
                )
            self.dataset = data_train
            self.eval_dataset = data_bench


    def format_prompt(self,
                      prompt: str,
                      system_prompt: str | None = None,
                      few_shot: List[Dict[str, Any]] | None = None
                      ) -> List[Dict[str, Any]]:
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        if few_shot:
            messages.extend(few_shot)
        messages.append({'role': 'user', 'content': prompt})
        return messages

    def format_dataset(self,
                       dataset: Dataset,
                       system_prompt: str | None = None,
                       few_shot: List[Dict[str, Any]] | None = None,
                       question_key: str = "question",
                       answer_key: str = "answer") -> Dataset:
        # Extract format_prompt as a standalone function to avoid capturing self
        def format_prompt_fn(prompt: str) -> List[Dict[str, Any]]:
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            if few_shot:
                messages.extend(few_shot)
            messages.append({'role': 'user', 'content': prompt})
            return messages

        if self.message_type == 'chat':
            if answer_key == "answer":
                return dataset.map(lambda x: {
                    "prompt": format_prompt_fn(x[question_key]),
                }, num_proc=self.max_concurrent)
            else:
                return dataset.map(lambda x: {
                    "prompt": format_prompt_fn(x[question_key]),
                    "answer": x[answer_key]
                }, num_proc=self.max_concurrent)

        # for completion, we expect 'prompt' and 'answer'
        return dataset

    def rollout(self, rollout: Rollout) -> Rollout:
        """
        Returns completion (str or message list) and null state.
        This is the generic QA rollout.
        :param rollout:
        """
        completion = self.sample(
            rollout=rollout,
        )
        if self.message_type == 'chat':
            rollout.samples = [{'role': 'assistant', 'content': completion}]
        else:
            rollout.samples = completion

        return rollout


    # def generate(self,
    #              inputs: Dict[str, List[Any]] | Dataset,
    #              client: OpenAI | None = None,
    #              model: str | None = None,
    #              sampling_args: Dict[str, Any] = {},
    #              max_concurrent: int | None = None,
    #              score_rollouts: bool = True,
    #              **kwargs: Any) -> Dict[str, Any]:
    #     """
    #     Generate completions and rewards for a given set of inputs.
    #     """
    #     # use class-level client and model if not provided
    #     if client is None:
    #         assert self.client is not None
    #         client = self.client
    #     if model is None:
    #         assert self.model is not None
    #         model = self.model
    #     gen_sampling_args = deepcopy(self.client_args)
    #     gen_sampling_args.update(sampling_args)
    #     if max_concurrent is None:
    #         max_concurrent = self.max_concurrent
    #
    #     # run rollouts
    #     if isinstance(inputs, Dataset):
    #         # get prompt column
    #         results = {col: deepcopy(inputs[col]) for col in inputs.column_names}
    #     else:
    #         results = deepcopy(inputs)
    #
    #     # Prepare items for run_rollouts
    #     items = []
    #     for i in range(len(results['prompt'])):
    #         item = {k: v[i] for k, v in results.items()}
    #         items.append(item)
    #
    #     rollouts = self.run_rollouts(
    #         items=items,
    #         client=client,
    #         model=model,
    #         sampling_args=gen_sampling_args,
    #         max_concurrent=max_concurrent,
    #         **kwargs
    #     )
    #     results['completion'] = [rollout[0] for rollout in rollouts]
    #     results['state'] = [rollout[1] for rollout in rollouts]
    #     if 'task' not in results:
    #         results['task'] = ['default'] * len(results['prompt'])
    #     if score_rollouts:
    #         results_rewards = self.rubric.score_rollouts(
    #             prompts=results['prompt'],
    #             completions=results['completion'],
    #             answers=results['answer'],
    #             states=results['state'],
    #             tasks=results['task'],
    #             max_concurrent=max_concurrent,
    #             apply_weights=True
    #         )
    #         results.update(results_rewards)
    #     return results

    # def evaluate(self,
    #              client: OpenAI | None = None,
    #              model: str | None = None,
    #              sampling_args: Dict[str, Any] = {},
    #              num_samples: int = -1,
    #              max_concurrent: int = 32,
    #              **kwargs: Any
    #             ) -> Dict[str, Any]:
    #     """
    #     Evaluate model on the Environment evaluation dataset.
    #     """
    #     # use class-level client and model if not provided
    #     if client is None:
    #         assert self.client is not None
    #         client = self.client
    #     if model is None:
    #         assert self.model is not None
    #         model = self.model
    #
    #     if self.eval_dataset is None:
    #         self.logger.info('eval_dataset is not set, falling back to train dataset')
    #         assert self.dataset is not None
    #         inputs = self.dataset
    #     else:
    #         inputs = self.eval_dataset
    #     if num_samples > 0:
    #         inputs = inputs.select(range(num_samples))
    #
    #     results = self.generate(
    #         inputs, client, model, sampling_args, max_concurrent, **kwargs
    #     )
    #     return results

    # def make_dataset(self,
    #                  results: Dict[str, Any] | None = None,
    #                  push_to_hub: bool = False,
    #                  hub_name: str | None = None,
    #                  client: OpenAI | None = None,
    #                  model: str | None = None,
    #                  max_concurrent: int | None = None,
    #                  num_samples: int = -1,
    #                  sampling_args: Dict[str, Any] = {'temperature': 0.6},
    #                  state_columns: List[str] = [],
    #                  extra_columns: List[str] = [],
    #                  **kwargs: Any) -> Dataset:
    #     """
    #     Make a dataset from the evaluation results.
    #     """
    #     if results is None and client is None:
    #         raise ValueError('Either results or client must be provided')
    #     if push_to_hub and hub_name is None:
    #         raise ValueError('hub_name must be provided if push_to_hub is True')
    #
    #     if results is None:
    #         # use class-level client and model if not provided
    #         if client is None:
    #             assert self.client is not None
    #             client = self.client
    #         if model is None:
    #             assert self.model is not None
    #             model = self.model
    #         if max_concurrent is None:
    #             max_concurrent = self.max_concurrent
    #         results = self.evaluate(
    #             client,
    #             model,
    #             sampling_args,
    #             num_samples,
    #             max_concurrent,
    #             **kwargs
    #         )
    #     cols = ['prompt', 'completion', 'answer', 'reward']
    #     if 'task' in results and results['task'][0] is not None:
    #         cols.append('task')
    #     if 'state' in results:
    #         for col in state_columns:
    #             if col in results['state'][0]:
    #                 results[col] = [state[col] for state in results['state']]
    #                 cols.append(col)
    #             else:
    #                 self.logger.warning(f'Column {col} not found in state, skipping from dataset.')
    #     for col in extra_columns:
    #         if col in results:
    #             cols.append(col)
    #         else:
    #             self.logger.warning(f'Column {col} not found in results, skipping from dataset.')
    #     dataset = Dataset.from_dict({
    #         col: results[col] for col in cols
    #     })
    #     if push_to_hub:
    #         assert hub_name is not None
    #         dataset.push_to_hub(hub_name)
    #     return dataset
