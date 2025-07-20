import asyncio
import concurrent.futures
import logging
import typing
from abc import ABC, abstractmethod
from asyncio import Semaphore
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple

from datasets import Dataset, IterableDataset  # type: ignore
from openai import OpenAI
from openai.types.chat import ChatCompletion

from errloom.aliases import Data
from errloom.utils.model_utils import load_data
from errloom.defaults import DATASET_MAX_CONCURRENT, DEFAULT_MAX_CONCURRENT
from errloom.interop.mock_client import MockClient
from errloom.tapestry import Rollout, Tapestry, Context
from errloom.utils.log import LogContext, indent_decorator
from errloom.utils import log

if typing.TYPE_CHECKING:
    import torch.nn
    import transformers

# noinspection PyDefaultArgument

class Loom(ABC):
    """
    The loom is a machine to process a batch of entry data.
    The weave function is called to weave the tapestry of
    rollout threads.

    The base weave function handles multi-thread processing of the
    different threads in parallel.

    TODO we should add LoomDaemon which runs a loom in the background and aims to refill a pool of rollouts to N that end users draw out of in batches of whatever desired size M
    Subclass implementations of the loom focus implement the rollout function for one rollout in isolation.
    Cross-rollout interaction is not possible, but happy to consider if there is a potent use case leading
    to nauseating levels of intelligence acceleration.
    """

    def __init__(self,
                 model: str | Tuple[str, 'torch.nn.Module'] | None = None,
                 tokenizer: str | Tuple[str, 'transformers.PreTrainedTokenizer'] | None = None,
                 client: OpenAI | None = None,
                 client_args: Dict[str, Any] = {},
                 message_type: Literal['chat', 'completion'] = 'chat',
                 data: Optional[Data | str] = None,
                 data_train: Optional[Data | str] = None,
                 data_bench: Optional[Data | str] = None,
                 data_split: Optional[float] = None,
                 max_concurrent: int = DEFAULT_MAX_CONCURRENT,
                 dry: bool = False):
        self.trainer: Optional['GRPOTrainer'] = None
        self.client = client or MockClient()
        self.client_args = {
            **client_args,
            'n':          1,  # n > 1 not supported; use duplicate prompts for multiple completions
            'extra_body': {
                'skip_special_tokens':           False,
                'spaces_between_special_tokens': False,
            }
        }
        self.message_type: Literal['chat', 'completion'] = message_type
        self.max_concurrent = max_concurrent
        self.dry = dry
        self.logger = log.getLogger(f'errloom.looms.{self.__class__.__name__}')
        self.init_data(data, data_bench, data_train, data_split)

        if isinstance(model, Tuple):
            self.model_name = model[0]
            self.model = model[1]
        elif isinstance(model, str):
            from errloom.utils.model_utils import get_model
            self.model_name = model
            self.model = get_model(model) if not dry else None

        if isinstance(tokenizer, Tuple):
            self.tokenizer = tokenizer[0]
            self.tokenizer_name = tokenizer[1]
        elif isinstance(tokenizer, str):
            from errloom.utils.model_utils import get_tokenizer
            self.tokenizer_name = model
            self.tokenizer = get_tokenizer(tokenizer) if not dry else None

        if not dry:
            with LogContext("👟 Initializing trainer...", "✓ Trainer ready", logger=self.logger):
                from errloom.training.grpo_trainer import GRPOTrainer
                from errloom.defaults import grpo_defaults
                assert self.model is not None
                assert self.tokenizer is not None
                assert self.model_name is not None
                self.trainer = GRPOTrainer(
                        model=self.model,  # type: ignore
                        tokenizer=self.tokenizer,  # type: ignore
                        args=grpo_defaults(name=self.model_name))


        valid_data = self.data or self.data_train and self.data_bench
        if not valid_data:
            raise ValueError('Either data or data_train & data_bench must be provided.')

        # TODO not sure what this is
        if client_args is not None and 'extra_body' in client_args:
            self.client_args['extra_body'].update(client_args['extra_body'])
        for k, v in client_args.items():
            if k != 'extra_body':
                self.client_args[k] = v

        # Ensure asyncio.to_thread doesn't hit default 32 thread limit
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
            loop.set_default_executor(executor)

    def init_data(self, data, data_bench, data_train, data_split: Optional[float] = 0.5):
        self.data = load_data(data)
        self.data_train = self.data if data == data_train else load_data(data_train or data)
        self.data_bench = self.data if data == data_bench else load_data(data_bench or data)

        # Split the base dataset so train & bench don't see the same thing (50:50 by default)
        if data_split is not None and data == data_train and data == data_bench:
            base = self.data
            if base is None:
                raise ValueError("Cannot split None dataset")
            if isinstance(base, Dataset):
                sz = len(base)
                sz1 = int(sz * data_split)
                data1 = base.select(range(sz1))
                data2 = base.select(range(sz1, sz))
                self.logger.error(f"[green]✓[/] Split {data_split:.1%}: {len(data1)} train, {len(data2)} eval")
            elif isinstance(base, IterableDataset):
                self.logger.error("[yellow]⚠ Dataset is iterable, taking first 1500 items.[/]")
                items = list(base.take(1500))
                split_idx = int(len(items) * data_split)
                data1 = Dataset.from_list(items[:split_idx])
                data2 = Dataset.from_list(items[split_idx:])
                self.logger.error(f"[green]✓[/] Split {data_split:.1%}: {len(data1)} train, {len(data2)} eval")
            else:
                raise TypeError(f"Cannot process dataset of type {type(base)}")


    @indent_decorator("ROLLOUT")
    @abstractmethod
    def rollout(self, roll: Rollout) -> Rollout:
        """
        Returns a tuple of (completion, state).
        """
        pass

    def take_data(self, n=None) -> Data:
        if self.data is None:
            raise ValueError("[red]Not enough data in the training set to perform a dry run.[/red]")
        data = list(self.data.take(n))
        if not data:
            raise ValueError("[red]Not enough data in the training set to perform a dry run.[/red]")
        if n is not None and len(data) < n:
            self.logger.warning(f"[yellow]Warning: Could only fetch {len(data)} samples for the dry run.[/yellow]")

        return Dataset.from_list(data)

    def weave(self, rows: Data | int) -> Tapestry:
        """
        Weave rollouts for the input dataset slice (batch of rows)
        Each row is unrolled through the run function.
        # TODO This used to take DataDict or dict[str, List[Any]] and im not down with that.
        # TODO convert to DataDict on the way in and let the linter blow up
        """

        # sample_args = deepcopy(self.client_args)  # TODO this feels poor performance for no reason
        # sample_args.update(sampling_args)  # TODO we should probably configure this on the env beforehand

        if rows is None or isinstance(rows, int):
            rows = self.take_data(rows)
        
        if isinstance(rows, int):
            raise ValueError("rows must be a Data object or an integer")

        assert isinstance(rows, Data)

        # self.logger.header_info("")
        self.logger.push_info("WEAVE")

        async def unroll_row(semaphore: Semaphore, state: Rollout) -> Rollout:
            """
            Run a rollout for a given prompt.
            Returns a tuple of (completion, state).
            """
            async with semaphore:
                return await asyncio.to_thread(self.rollout, state)

        async def unroll_rows() -> List[Rollout]:
            """
            Run rollouts for a given list of prompts and return the completions.
            """
            semaphore = Semaphore(self.max_concurrent)
            tasks = []
            rolls = []

            for row in rows:
                roll = Rollout(dict(row), sampling_args=self.client_args)
                rolls.append(roll)
                tasks.append(unroll_row(semaphore, roll))

            self.logger.info(f'Running {len(tasks)} rollouts')
            for f in asyncio.as_completed(tasks):
                await f

            return rolls

        # Run the async function in a new event loop
        # This avoids issues with nested event loops
        tapestry = asyncio.run(unroll_rows())
        tapestry = Tapestry(tapestry)
        tapestry.max_concurrent = self.max_concurrent

        if self.dry:
            self.logger.info(f"Received {len(tapestry.rollouts)} rollouts:")
            for roll in tapestry.rollouts:
                self.logger.info(f"1. {roll}")

        self.logger.pop()
        return tapestry

    def get_train_data(self, n: int = -1, seed: int = 0) -> Data | None:
        if n > 0 and self.data_train is not None:
            return self.data_train.shuffle(seed=seed).select(range(n))  # type: ignore
        return self.data_train

    def get_bench_data(self, n: int = -1, seed: int = 0) -> Data | None:
        if n > 0 and self.data_bench is not None:
            return self.data_bench.shuffle(seed=seed).select(range(n))  # type: ignore
        return self.data_bench

    # def get_rule_funcs(self) -> List[FnRule]:
    #     return self.attractor.rule_funcs
    #
    # def get_rule_weights(self) -> List[float]:
    #     return self.attractor.rule_weights

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


    def test(self, num_samples: int = -1) -> Tapestry:
        """
        Evaluate model on the Environment evaluation dataset.
        TODO this needs better definition... like, what exactly is this doing? running one round of inference? one batch? etc. we can see that it calls generate, and this should be better emphasized somehow
        """
        # use class-level client and model if not provided
        assert self.client is not None
        assert self.model_name is not None

        if self.data_bench is None:
            self.logger.info('eval_dataset is not set, falling back to train dataset')
            assert self.data_train is not None
            inputs = self.data_train
        else:
            inputs = self.data_bench
        if num_samples > 0:
            inputs = inputs.select(range(num_samples))

        return self.weave(inputs)

    def sample(self, rollout: Rollout, sanitize_sampling_args: bool = True) -> str:
        """
        Sample a model response for a given prefix context
        MessageList or raw str in completion mode

        Convenience function for wrapping (chat, completion) API calls.
        Returns special error messages for context length issues.
        """
        client = self.client
        model = self.model_name

        if sanitize_sampling_args:
            sanitized_args = self.sanitize_sampling_args(client, rollout.sampling_args)
        else:
            sanitized_args = rollout.sampling_args

        def _sample() -> str:
            try:
                res: ChatCompletion
                if self.message_type == 'chat':
                    assert isinstance(rollout.context.messages, list)
                    # Convert messages to the format expected by OpenAI API
                    messages = []
                    for msg in rollout.context.messages:
                        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                            messages.append({
                                'role': msg['role'],
                                'content': msg['content']
                            })
                    res = client.chat.completions.create(model=model, messages=messages, **sanitized_args)
                    # Check if generation was truncated due to max_tokens
                    if res.choices[0].finish_reason == 'length':
                        return "[ERROR] max_tokens_reached"
                    content = res.choices[0].message.content
                    return content if content is not None else "[ERROR] empty_response"
                elif self.message_type == 'completion':
                    assert isinstance(rollout.context.text, str)
                    from openai.types.completion import Completion
                    completion_res: Completion = client.completions.create(model=model, prompt=rollout.context.text, **sanitized_args)
                    # Check if generation was truncated due to max_tokens
                    if completion_res.choices[0].finish_reason == 'length':
                        return "[ERROR] max_tokens_reached"
                    content = completion_res.choices[0].text
                    return content if content is not None else "[ERROR] empty_response"

                raise ValueError(f"Unsupported message_type: {self.message_type}")
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
                     tapestry: Tapestry | None = None,
                     push_to_hub: bool = False,
                     hub_name: str | None = None,
                     num_samples: int = -1,
                     state_columns: List[str] = [],
                     extra_columns: List[str] = []) -> Data:
        """
        Make a dataset from the evaluation results.
        """
        if tapestry is None and self.client is None:
            raise ValueError('Either tapestry or client must be provided')
        if push_to_hub and hub_name is None:
            raise ValueError('hub_name must be provided if push_to_hub is True')

        if tapestry is None:
            assert self.client is not None
            assert self.model_name is not None
            tapestry = self.test(num_samples)

        # Use the Tapestry method to extract data from rollouts
        data_dict = tapestry.to_dataset(
            state_columns=state_columns, 
            extra_columns=extra_columns
        )

        dataset = Dataset.from_dict(data_dict)
        if push_to_hub:
            assert hub_name is not None
            dataset.push_to_hub(hub_name)
        return dataset

    def format_dataset_qa(self, system: str, few_shot: Optional[str] = None):
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

        dtrain = self.get_train_data()
        dbench = self.get_bench_data()

        if self.message_type == 'chat':
            if dtrain is not None:
                self.data_train = _format(dtrain, system)
            else:
                self.data_train = None
            if dbench is not None:
                self.data_bench = _format(dbench, system)
            else:
                self.data_bench = None
        else:
            if system or few_shot:
                raise ValueError(
                    'The fields "system_prompt" and "few_shot" are not supported for completion tasks.'
                    'Please use message_type="chat" instead, or pre-format your dataset '
                    'to contain "prompt" and "answer" columns.'
                )
            self.data_train = dtrain
            self.data_bench = dbench

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
