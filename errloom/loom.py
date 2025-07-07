import asyncio
import concurrent.futures
import logging
from abc import ABC, abstractmethod
from asyncio import Semaphore
from copy import deepcopy
from typing import Any, Dict, List, Literal

from openai import OpenAI
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

from errloom.aliases import Data
from errloom.attractor import Attractor, FnRule
from errloom.interop.mock_client import MockClient
from errloom.parsers.parser import Parser
from errloom.rollout import Rollout, Rollouts

DEFAULT_MAX_CONCURRENT = 512
DATASET_MAX_CONCURRENT = 32

logger = logging.getLogger(__file__)

# noinspection PyDefaultArgument
class Loom(ABC):
    """
    Base class for all looms.
    A loom generates threads and tapestries of text.
    """

    def __init__(self,
                 client: OpenAI | None = None,
                 model: str | None = None,
                 roll_dataset: Data | None = None,
                 eval_dataset: Data | None = None,
                 system_prompt: str | None = None,
                 parser: Parser = Parser(),
                 attractor: Attractor = Attractor(),
                 sampling_args: Dict[str, Any] = {},
                 max_concurrent: int = DEFAULT_MAX_CONCURRENT,
                 message_type: Literal['chat', 'completion'] = 'chat',
                 dry: bool = False,
                 **kwargs: Any):
        self.client = client or MockClient()
        self.model = model or "mock-model"
        self.message_type: Literal['chat', 'completion'] = message_type
        self.system_prompt = system_prompt  # optional; environments can use this or do their own thing
        self.max_concurrent = max_concurrent
        self.dataset = roll_dataset
        self.eval_dataset = eval_dataset
        self.parser = parser
        self.attractor = attractor
        self.dry = dry

        # Ensure asyncio.to_thread doesn't hit default 32 thread limit
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
            loop.set_default_executor(executor)

        self.client_args = {
            'n':          1,  # n > 1 not supported; use duplicate prompts for multiple completions
            'extra_body': {
                'skip_special_tokens':           False,
                'spaces_between_special_tokens': False,
            },
        }
        if sampling_args is not None and 'extra_body' in sampling_args:
            self.client_args['extra_body'].update(sampling_args['extra_body'])
        for k, v in sampling_args.items():
            if k != 'extra_body':
                self.client_args[k] = v
        self.logger = logging.getLogger(f'errloom.looms.{self.__class__.__name__}')
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.dataset is None and self.eval_dataset is None:
            raise ValueError('Either dataset or eval_dataset must be provided')

    @abstractmethod
    def run(self, state: Rollout) -> Rollout:
        """
        Run a rollout for a given prompt.
        Returns a tuple of (completion, state).
        """
        pass

    def get_dataset(self, n: int = -1, seed: int = 0) -> Data | None:
        if n > 0 and self.dataset is not None:
            return self.dataset.shuffle(seed=seed).select(range(n))  # type: ignore
        return self.dataset

    def get_eval_dataset(self, n: int = -1, seed: int = 0) -> Data | None:
        if n > 0 and self.eval_dataset is not None:
            return self.eval_dataset.shuffle(seed=seed).select(range(n))  # type: ignore
        return self.eval_dataset

    def get_rule_funcs(self) -> List[FnRule]:
        return self.attractor.rule_funcs

    def get_rule_weights(self) -> List[float]:
        return self.attractor.rule_weights

    def sanitize_sampling_args(self, client: OpenAI, sampling_args: Dict[str, Any]) -> Dict[str, Any]:
        from urllib.parse import urlparse
        url = urlparse(str(client.base_url))
        # check if url is not localhost/127.0.0.1/0.0.0.0
        if url.netloc not in ["localhost", "127.0.0.1", "0.0.0.0"]:
            sanitized_args = deepcopy(sampling_args)
            # remove extra_body
            sanitized_args.pop('extra_body', None)
            return sanitized_args
        return sampling_args

    def unroll(self, rows: Data) -> Rollouts:
        """
        Generate rollouts for the input dataset slice (batch of rows)
        Each row is unrolled through the run function.
        # TODO This used to take DataDict or dict[str, List[Any]] and im not down with that.
        # TODO convert to DataDict on the way in and let the linter blow up
        """

        # sample_args = deepcopy(self.client_args)  # TODO this feels poor performance for no reason
        # sample_args.update(sampling_args)  # TODO we should probably configure this on the env beforehand

        # if 'task' not in bag.column_names:
        #     bag = bag.add_column('task', ['default'] * len(bag))  # type: ignore
        # if 'info' not in bag.column_names:
        #     bag = bag.add_column('info', ['default'] * len(bag))  # type: ignore

        def setup_executor(loop):
            if loop._default_executor is None:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent)
                loop.set_default_executor(executor)

        async def run_row(semaphore: Semaphore, state: Rollout) -> Rollout:
            """
            Run a rollout for a given prompt.
            Returns a tuple of (completion, state).
            """
            async with semaphore:
                return await asyncio.to_thread(self.run, state)

        async def run_rows() -> List[Rollout]:
            """
            Run rollouts for a given list of prompts and return the completions.
            """
            semaphore = Semaphore(self.max_concurrent)
            rollout_tasks = []
            rollouts = []

            for row in rows:
                rollout = Rollout(dict(row), sampling_args=self.client_args, dry=self.dry)
                rollouts.append(rollout)
                rollout_tasks.append(run_row(semaphore, rollout))

            logger.info(f'Running {len(rollout_tasks)} rollouts')
            for f in asyncio.as_completed(rollout_tasks):
                await f

            return rollouts

        coro = run_rows()
        loop = asyncio.new_event_loop()
        setup_executor(loop)
        asyncio.set_event_loop(loop)
        # TODO this used to be try/catch, but do we really want to fail safely here? are there cases where it would randomly fail for just one random instance? usually it's better to let it blow up early and make the code more stable for the future
        rollouts = Rollouts(loop.run_until_complete(coro))
        loop.close()
        asyncio.set_event_loop(None)
        # try:
        # except RuntimeError:
        #     # Jupyter notebook or existing event loop
        #     import nest_asyncio
        #     nest_asyncio.apply()
        #     loop = asyncio.get_running_loop()
        #     # noinspection PyTypeChecker
        #     setup_executor(loop)
        #     rollouts = Rollouts(loop.run_until_complete(coro))

        rollouts.max_concurrent = self.max_concurrent
        self.attractor.feels(rollouts)

        # ---------------------------------------

        print(f"Received {len(rollouts.rollouts)} rollouts:")
        for rollout in rollouts.rollouts:
            print("")
            print(rollout)

        print("")
        print("-" * 60)
        print()

        return rollouts

    def test(self, num_samples: int = -1) -> Rollouts:
        """
        Evaluate model on the Environment evaluation dataset.
        TODO this needs better definition... like, what exactly is this doing? running one round of inference? one batch? etc. we can see that it calls generate, and this should be better emphasized somehow
        """
        # use class-level client and model if not provided
        assert self.client is not None
        assert self.model is not None

        if self.eval_dataset is None:
            self.logger.info('eval_dataset is not set, falling back to train dataset')
            assert self.dataset is not None
            inputs = self.dataset
        else:
            inputs = self.eval_dataset
        if num_samples > 0:
            inputs = inputs.select(range(num_samples))

        return self.unroll(inputs)

    def sample(self, rollout: Rollout, sanitize_sampling_args: bool = True) -> str:
        """
        Sample a model response for a given prefix context
        MessageList or raw str in completion mode

        Convenience function for wrapping (chat, completion) API calls.
        Returns special error messages for context length issues.
        """
        client = self.client
        model = self.model

        if sanitize_sampling_args:
            sanitized_args = self.sanitize_sampling_args(client, rollout.sampling_args)
        else:
            sanitized_args = rollout.sampling_args

        def _sample():
            try:
                if self.message_type == 'chat':
                    assert isinstance(rollout.context.messages, list)
                    response = client.chat.completions.create(model=model, messages=rollout.context.messages, **sanitized_args)
                    # Check if generation was truncated due to max_tokens
                    if response.choices[0].finish_reason == 'length':
                        return "[ERROR] max_tokens_reached"
                    return response.choices[0].message.content  # type: ignore
                elif self.message_type == 'completion':
                    assert isinstance(rollout.context.text, str)
                    response = client.completions.create(model=model, prompt=rollout.context.text, **sanitized_args)
                    # Check if generation was truncated due to max_tokens
                    if response.choices[0].finish_reason == 'length':
                        return "[ERROR] max_tokens_reached"
                    return response.choices[0].text  # type: ignore

                raise ValueError
            except Exception as e:
                # Check for prompt too long errors
                error_msg = str(e)
                if "longer than the maximum" in error_msg or "exceeds the model" in error_msg:
                    return "[ERROR] prompt_too_long"
                # Re-raise other errors
                raise

        ret = _sample()
        rollout.samples.append(ret)
        return ret

    def make_dataset(self,
                     results: Dict[str, Any] | None = None,
                     push_to_hub: bool = False,
                     hub_name: str | None = None,
                     num_samples: int = -1,
                     state_columns: List[str] = [],
                     extra_columns: List[str] = []) -> Data:
        """
        Make a dataset from the evaluation results.
        """
        if results is None and self.client is None:
            raise ValueError('Either results or client must be provided')
        if push_to_hub and hub_name is None:
            raise ValueError('hub_name must be provided if push_to_hub is True')

        if results is None:
            assert self.client is not None
            assert self.model is not None
            results = self.test(num_samples)

        cols = ['prompt', 'completion', 'answer', 'gravity']
        if results['task'][0] is not None:
            cols.append('task')
        if 'state' in results:
            for col in state_columns:
                if col in results['state'][0]:
                    results[col] = [state[col] for state in results['state']]
                    cols.append(col)
                else:
                    self.logger.warning(f'Column {col} not found in state, skipping from dataset.')
        for col in extra_columns:
            if col in results:
                cols.append(col)
            else:
                self.logger.warning(f'Column {col} not found in results, skipping from dataset.')
        dataset = Data.from_dict({col: results[col] for col in cols})
        if push_to_hub:
            assert hub_name is not None
            dataset.push_to_hub(hub_name)
        return dataset

    def format_dataset_qa(self, few_shot=None):
        """
        Transforms the dataset to [prompt, answer] where the prompt is the entire
        pre-baked context prefix up to the model's point where it should run inference.
        This used to be the default into Environment, which is a symptom of being
        mindcontrolled by the government to decelerate open-source AI. We must
        forever strive towards greater generalization and modularity of features,
        explicit on/off switches and lego bricks.

        For chat message type:
        - Takes the question from the dataset and wraps it in a messages list format
        - Includes system prompt and few-shot examples if provided
        - Messages list contains dicts with 'role' and 'content' keys

        For completion message type:
        - Uses dataset as-is, but validates no system_prompt or few_shot are provided
        - Expects dataset to already have "prompt" and "answer" columns

        The formatted dataset is used by the environment to generate responses in a
        consistent format regardless of the underlying model API.

        TODO doccument this few_show argument - wtf is that?
        """

        def _format(dataset: Data,
                    system_prompt: str | None = None,
                    question_key: str = "question",
                    answer_key: str = "answer") -> Data:
            # skip if "prompt" already exists
            if "prompt" in dataset.column_names:
                return dataset

            # extract format_prompt as a standalone function to avoid capturing self
            def format_prompt_fn(prompt: str) -> List[Dict[str, Any]]:
                messages = []
                if system_prompt:
                    messages.append({'role': 'system', 'content': system_prompt})
                if few_shot:
                    messages.extend(few_shot)
                messages.append({'role': 'user', 'content': prompt})
                return messages

            if answer_key == "answer":
                return dataset.map(lambda x: {
                    "prompt": format_prompt_fn(x[question_key]),
                }, num_proc=min(self.max_concurrent, DATASET_MAX_CONCURRENT))
            else:
                return dataset.map(lambda x: {
                    "prompt": format_prompt_fn(x[question_key]),
                    "answer": x[answer_key]
                }, num_proc=min(self.max_concurrent, DATASET_MAX_CONCURRENT))

        dataset = self.dataset
        eval_dataset = self.eval_dataset

        if self.message_type == 'chat':
            if dataset is not None:
                self.dataset = _format(dataset, self.system_prompt)
            else:
                self.dataset = None
            if eval_dataset is not None:
                self.eval_dataset = _format(eval_dataset, self.system_prompt)
            else:
                self.eval_dataset = None
        else:
            if self.system_prompt or few_shot:
                raise ValueError(
                    'The fields "system_prompt" and "few_shot" are not supported for completion tasks.'
                    'Please use message_type="chat" instead, or pre-format your dataset '
                    'to contain "prompt" and "answer" columns.'
                )
            self.dataset = dataset
            self.eval_dataset = eval_dataset

    # def process_item(self,
    #                  item: Any,
    #                  client: OpenAI,
    #                  model: str,
    #                  sampling_args: Dict[str, Any] = {},
    #                  **kwargs: Any) -> Any:
    #     """
    #     Process a single item. Can be implemented by subclasses for custom generation logic.
    #     By default, this calls the standard rollout method.
    #     This is a synchronous method that will be run in a thread.
    #     """
    #     if 'prompt' not in item or 'answer' not in item:
    #         raise ValueError("The default `process_item` requires 'prompt' and 'answer' in the item dict.")
    #
    #     return self.rollout(
    #         client=client,
    #         model=model,
    #         context=item['prompt'],
    #         answer=item['answer'],
    #         task=item.get('task', 'default'),
    #         info=item.get('info', {}),
    #         sampling_args=sampling_args,
    #         **kwargs
    #     )
    #
    # async def _process_single_item(self,
    #                                semaphore: Semaphore,
    #                                item: Any,
    #                                client: OpenAI,
    #                                model: str,
    #                                sampling_args: Dict[str, Any] = {},
    #                                **kwargs: Any) -> EnvOutput:
    #     """
    #     Run processing for a single item.
    #     """
    #     async with semaphore:
    #         return await asyncio.to_thread(self.process_item, item, client=client, model=model, sampling_args=sampling_args, **kwargs)
    #
    # async def _run_all_items(self,
    #                          items: List[Any],
    #                          client: OpenAI,
    #                          model: str,
    #                          sampling_args: Dict[str, Any] = {},
    #                          max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    #                          **kwargs: Any) -> List[EnvOutput]:
    #     """
    #     Run processing for a given list of items and return the results.
    #     """
    #     from tqdm.asyncio import tqdm_asyncio
    #     semaphore = Semaphore(max_concurrent)
    #     tasks = [
    #         self._process_single_item(semaphore, item, client, model, sampling_args, **kwargs)
    #         for item in items
    #     ]
    #     return await tqdm_asyncio.gather(
    #         *tasks,
    #         total=len(items),
    #         desc=f'Running {len(items)} items'
    #     )
    #
    # def run_processing(self,
    #                  items: List[Any],
    #                  client: OpenAI,
    #                  model: str,
    #                  sampling_args: Dict[str, Any] = {},
    #                  max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    #                  **kwargs: Any) -> list[EnvOutput]:
    #     """
    #     Run processing for a given list of items and return the results.
    #     """
    #     def setup_executor(loop):
    #         if loop._default_executor is None:
    #             executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
    #             loop.set_default_executor(executor)
    #
    #     coroutine = self._run_all_items(
    #         items=items,
    #         client=client,
    #         model=model,
    #         sampling_args=sampling_args,
    #         max_concurrent=max_concurrent,
    #         **kwargs
    #     )
    #     try:
    #         # Create new event loop with custom executor
    #         loop = asyncio.new_event_loop()
    #         setup_executor(loop)
    #         asyncio.set_event_loop(loop)
    #         try:
    #             return loop.run_until_complete(coroutine)
    #         finally:
    #             loop.close()
    #             asyncio.set_event_loop(None)
    #     except RuntimeError:
    #         # Jupyter notebook or existing event loop
    #         import nest_asyncio
    #         nest_asyncio.apply()
    #         loop = asyncio.get_running_loop()
    #         setup_executor(loop)
    #         return loop.run_until_complete(coroutine)
