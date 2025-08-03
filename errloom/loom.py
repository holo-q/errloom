import asyncio
import concurrent.futures
import typing
from abc import ABC, abstractmethod
from asyncio import Semaphore
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple


from errloom.aliases import Data
from errloom.defaults import DATASET_MAX_CONCURRENT, DEFAULT_MAX_CONCURRENT
from errloom.tapestry import Rollout, Tapestry
from errloom.lib import log
from errloom.lib.log import ContextAwareThreadPoolExecutor, PrintedText, indent_decorator, LogContext

if typing.TYPE_CHECKING:
    import torch.nn
    import transformers
    from errloom.session import Session
    from openai import OpenAI
    from openai.types.chat import ChatCompletion

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
                 client: Optional['OpenAI'] = None,
                 client_args: Dict[str, Any] = {},
                 message_type: Literal['chat', 'completion'] = 'chat',
                 data: Optional[Data | str] = None,
                 data_train: Optional[Data | str] = None,
                 data_bench: Optional[Data | str] = None,
                 data_split: Optional[float] = None,
                 max_concurrent: int = DEFAULT_MAX_CONCURRENT,
                 dry: bool = False,
                 unsafe: bool = False,
                 show_rollout_errors: bool = False,
                 session: Optional['Session'] = None,
                 dump_rollouts: Optional[str | bool] = False):
        # Import errlargs for dry training logic
        from errloom import errlargs # argp depends on loom to assign a default

        self.trainer: Optional['RLTrainer'] = None

        # Use provided client or create default MockClient
        if client is not None:
            self.client = client
        else:
            from errloom.interop.mock_client import MockClient
            self.client = MockClient()
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
        self.unsafe = unsafe
        self.show_rollout_errors = show_rollout_errors
        self.session = session
        self.dump_rollouts = dump_rollouts
        self.logger = log.getLogger(f'errloom.looms.{self.__class__.__name__}')
        self.init_data(data, data_bench, data_train, data_split)
        if isinstance(model, Tuple):
            self.model_name = model[0]
            self.model = model[1]
        elif isinstance(model, str):
            self.model_name = model
            # Load model even in dry mode for dry training
            need_model_for_dry_training = (dry and errlargs.command == "train" and errlargs.dry)
            if not dry or need_model_for_dry_training:
                from errloom.utils.model_utils import get_model
                self.model = get_model(model)
            else:
                self.model = None
        else:
            self.model_name = None
            self.model = None

        if isinstance(tokenizer, Tuple):
            self.tokenizer = tokenizer[0]
            self.tokenizer_name = tokenizer[1]
        elif isinstance(tokenizer, str):
            self.tokenizer_name = model
            # Load tokenizer even in dry mode for dry training
            need_tokenizer_for_dry_training = (dry and errlargs.command == "train" and errlargs.dry)
            if not dry or need_tokenizer_for_dry_training:
                from errloom.utils.model_utils import get_tokenizer
                self.tokenizer = get_tokenizer(tokenizer)
            else:
                self.tokenizer = None
        else:
            self.tokenizer_name = None
            self.tokenizer = None

        if not dry:
            assert self.model is not None
            assert self.tokenizer is not None
            assert self.model_name is not None
            with LogContext("👟 Initializing GRPO...", "GRPO init", logger=self.logger):
                from errloom.training.rl_trainer import RLTrainer
                from errloom.defaults import grpo_defaults, grpo_local_test_defaults, grpo_micro_test_defaults, grpo_cpu_test_defaults

                # Choose config based on testing flags
                if errlargs.cpu:
                    config = grpo_cpu_test_defaults(name=self.model_name)
                elif errlargs.micro_test:
                    config = grpo_micro_test_defaults(name=self.model_name)
                elif errlargs.local_test:
                    config = grpo_local_test_defaults(name=self.model_name)
                else:
                    config = grpo_defaults(name=self.model_name)

                self.trainer = RLTrainer(
                    loom=self,  # Pass self as the loom argument
                    model=self.model,  # type: ignore
                    tokenizer=self.tokenizer,  # type: ignore
                    args=config)
        else:
            # Check if we need a trainer for dry training mode
            if errlargs.command == "train" and errlargs.dry:
                # Create trainer in dry mode for dry training
                if self.model is not None and self.tokenizer is not None and self.model_name is not None:
                    with LogContext("🧪 Initializing Dry GRPO...", "dry_grpo_init", logger=self.logger):
                        from errloom.training.rl_trainer import RLTrainer
                        from errloom.defaults import grpo_defaults, grpo_local_test_defaults, grpo_micro_test_defaults, grpo_cpu_test_defaults

                        # Choose config based on testing flags
                        if errlargs.cpu:
                            config = grpo_cpu_test_defaults(name=self.model_name)
                        elif errlargs.micro_test:
                            config = grpo_micro_test_defaults(name=self.model_name)
                        elif errlargs.local_test:
                            config = grpo_local_test_defaults(name=self.model_name)
                        else:
                            config = grpo_defaults(name=self.model_name)

                        self.trainer = RLTrainer(
                            loom=self,  # Pass self as the loom argument
                            model=self.model,  # type: ignore
                            tokenizer=self.tokenizer,  # type: ignore
                            args=config,
                            dry=True)  # Pass dry=True to trainer

        valid_data = self.data or self.data_train and self.data_bench
        if not valid_data:
            raise ValueError('Either data or data_train & data_bench must be provided.')

        # TODO not sure what this is
        if client_args is not None and 'extra_body' in client_args:
            self.client_args['extra_body'].update(client_args['extra_body'])
        for k, v in client_args.items():
            if k != 'extra_body':
                self.client_args[k] = v

    def init_data(self, data, data_bench, data_train, data_split: Optional[float] = 0.5):
        from datasets import Dataset, IterableDataset  # type: ignore
        from errloom.utils.data_utils import load_data
        self.data = load_data(data)
        self.data_train = self.data if data == data_train or data_train is None else load_data(data_train or data)
        self.data_bench = self.data if data == data_bench or data_bench is None else load_data(data_bench or data)

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

    def invoke(self, state: Rollout) -> Rollout:
        """
        Invoke the rollout with appropriate error handling based on unsafe setting.
        """
        # TODO we should move this to argp.invoke to make it reusable based on its own internal unsafe flag
        # Simply call the rollout method - the context inheritance is handled in unroll_row
        if self.unsafe:
            return self.rollout(state)
        else:
            try:
                return self.rollout(state)
            except Exception as e:
                # Log full stacktrace to file only
                from errloom.lib.log import log_stacktrace_to_file_only
                import logging
                log_stacktrace_to_file_only(logging.getLogger(), e, f"rollout {id(state)}")

                # Add error information to the rollout
                state.samples.append(f"[ERROR] {type(e).__name__}: {str(e)}")
                state.extra['error'] = {
                    'type':       type(e).__name__,
                    'message':    str(e),
                    'rollout_id': id(state)
                }

                # Log a brief error message to console (truncated for readability)
                # Only show in console if explicitly requested
                if self.show_rollout_errors or True:
                    error_msg = str(e)
                    if len(error_msg) > 100:
                        error_msg = error_msg[:97] + "..."
                    self.logger.warning(f"[red]Rollout failed: {type(e).__name__}: {error_msg}[/red]")

                return state

    def take_data(self, n=None) -> Data:
        from datasets import Dataset
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

        from errloom.aliases import is_data_type
        assert is_data_type(rows), f"Expected Data type, got {type(rows)}"

        # self.logger.header_info("")
        self.logger.push_info("WEAVE")

        async def unroll_row(semaphore: Semaphore, state: Rollout, executor: concurrent.futures.Executor) -> Rollout:
            """
            Run a rollout for a given prompt.
            The logging context is automatically propagated by ContextAwareThreadPoolExecutor.
            """
            async with semaphore:
                # Create a thread with a meaningful name for better logging
                def invoke_with_context():
                    import threading
                    # Set thread name for better logging
                    current_thread = threading.current_thread()
                    if not current_thread.name.startswith('ThreadPool'):
                        current_thread.name = f"Worker-{id(state)}"
                    return self.invoke(state)

                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(executor, invoke_with_context)

        async def unroll_rows(executor: concurrent.futures.Executor) -> List[Rollout]:
            """
            Run rollouts for a given list of prompts and return the completions.
            """
            semaphore = Semaphore(self.max_concurrent)
            tasks = []
            rolls = []

            for row in rows:
                roll = Rollout(dict(row), sampling_args=self.client_args)
                rolls.append(roll)
                tasks.append(unroll_row(semaphore, roll, executor))

            self.logger.info(f'Running {len(tasks)} rollouts')
            for f in asyncio.as_completed(tasks):
                await f

            return rolls

        # Run the async function in a new event loop
        # This avoids issues with nested event loops
        executor = ContextAwareThreadPoolExecutor(max_workers=self.max_concurrent)
        tapestry = asyncio.run(unroll_rows(executor))
        tapestry = Tapestry(tapestry)
        tapestry.max_concurrent = self.max_concurrent

        if self.dry:
            self.logger.info(f"Received {len(tapestry.rollouts)} rollouts:")
            for i, roll in enumerate(tapestry.rollouts):
                self.logger.info_hl(f"{i + 1}. {log.to_json_text(roll, indent=2, redactions=['contexts'])}")
                self.logger.info(PrintedText(roll.to_rich()))

        # Dump rollouts to session if requested
        if self.dump_rollouts and self.session:
            self._dump_rollouts_to_session(tapestry)

        self.logger.pop()
        return tapestry

    def _dump_rollouts_to_session(self, tapestry: Tapestry) -> None:
        """
        Save rollouts to the session directory as flat text files suitable for training.
        If a rollout has multiple contexts, split it into multiple files with proper indexing.

        Args:
            tapestry: The tapestry containing rollouts to dump
        """
        if not self.session or not self.dump_rollouts:
            return

        from datetime import datetime

        # Determine subdirectory name
        if isinstance(self.dump_rollouts, str):
            subdir_name = self.dump_rollouts
        else:
            # Use timestamp as default subdirectory name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            subdir_name = f"rollouts_{timestamp}"

        # Create rollouts directory in session
        rollouts_dir = self.session.dirpath / subdir_name
        rollouts_dir.mkdir(exist_ok=True, parents=True)

        total_files = 0

        # Process each rollout
        for rollout_idx, rollout in enumerate(tapestry.rollouts):
            if not rollout.contexts:
                # No contexts, create a simple text file with samples
                sample_strings = rollout.get_sample_strings() if hasattr(rollout, 'get_sample_strings') else []
                content = "\n".join(sample_strings) if sample_strings else ""
                if content.strip():
                    file_path = rollouts_dir / f"rollout_{rollout_idx:04d}.txt"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    total_files += 1
                continue

            # Handle multiple contexts by splitting into separate files
            for context_idx, context in enumerate(rollout.contexts):
                content_parts = []

                # Extract text content from context
                if context.text:
                    content_parts.append(context.text)

                # Extract messages as conversation format
                if context.messages:
                    for msg in context.messages:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        if content.strip():
                            content_parts.append(f"{role}: {content}")

                # Add samples for this rollout (only once, on the last context)
                if context_idx == len(rollout.contexts) - 1 and rollout.samples:
                    for sample in rollout.samples:
                        if sample.strip():
                            content_parts.append(f"assistant: {sample}")

                # Write content if we have any
                if content_parts:
                    content = "\n\n".join(content_parts)

                    if len(rollout.contexts) == 1:
                        # Single context - use simple naming
                        file_path = rollouts_dir / f"rollout_{rollout_idx:04d}.txt"
                    else:
                        # Multiple contexts - include context index
                        file_path = rollouts_dir / f"rollout_{rollout_idx:04d}_ctx_{context_idx:02d}.txt"

                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    total_files += 1

        # Create a metadata file with training information
        metadata_file = rollouts_dir / "metadata.txt"
        metadata_content = f"""# Training Data Metadata
Generated: {datetime.now().isoformat()}
Loom: {self.__class__.__name__}
Total rollouts: {len(tapestry.rollouts)}
Total files: {total_files}
Max concurrent: {tapestry.max_concurrent}

# File naming convention:
# rollout_NNNN.txt - Single context rollout
# rollout_NNNN_ctx_MM.txt - Multiple context rollout (context MM of rollout NNNN)

# Content format:
# - Conversation turns separated by double newlines
# - Format: "role: content"
# - Roles: system, user, assistant, etc.
"""

        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(metadata_content)

        self.logger.info(f"[green]✓[/] Dumped {len(tapestry.rollouts)} rollouts as {total_files} text files to {rollouts_dir}")

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

    def sanitize_sampling_args(self, client: 'OpenAI', sampling_args: Dict[str, Any]) -> Dict[str, Any]:
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


    def _handle_api_response(self, response, message_type: str) -> str:
        """Handle API response and extract content with error checking."""
        if response.choices[0].finish_reason == 'length':
            return "[ERROR] max_tokens_reached"

        if message_type == 'chat':
            content = response.choices[0].message.content
        else:  # completion
            content = response.choices[0].text

        return content if content is not None else "[ERROR] empty_response"

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

        try:
            res: ChatCompletion
            if self.message_type == 'chat':
                # Use Rollout method to get OpenAI format messages
                messages = rollout.to_api_chat()
                res = client.chat.completions.create(model=model, messages=messages, **sanitized_args)  # type: ignore
                ret = self._handle_api_response(res, self.message_type)
            elif self.message_type == 'completion':
                # Use Rollout method to get completion text
                prompt = rollout.to_text()
                from openai.types.completion import Completion
                completion_res: Completion = client.completions.create(model=model, prompt=prompt, **sanitized_args)
                ret = self._handle_api_response(completion_res, self.message_type)
            else:
                raise ValueError(f"Unsupported message_type: {self.message_type}")
        except Exception as e:
            error_msg = str(e)
            if "longer than the maximum" in error_msg or "exceeds the model" in error_msg:
                return "[ERROR] prompt_too_long"
            raise

        # Convert response to proper message format for new architecture
        if hasattr(rollout, 'samples') and isinstance(rollout.samples, list):
            # For new architecture, always store as message dictionaries
            rollout.samples.append({'role': 'assistant', 'content': ret})
        else:
            # Fallback for legacy compatibility
            if not hasattr(rollout, 'samples'):
                rollout.samples = []
            rollout.samples.append({'role': 'assistant', 'content': ret})

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
        from datasets import Dataset  # type: ignore

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

    def _format_prompt_for_chat(self, prompt: str, system_prompt: str | None = None, few_shot: str | List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
        """Format a prompt for chat API with system prompt and few-shot examples."""
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        if few_shot:
            if isinstance(few_shot, str):
                # Parse few_shot string if needed, for now just skip
                pass
            else:
                messages.extend(few_shot)
        messages.append({'role': 'user', 'content': prompt})
        return messages

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
                return self._format_prompt_for_chat(prompt, system_prompt, few_shot)

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
