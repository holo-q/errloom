import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from errloom.loom import Loom
from errloom.tapestry import Rollout, Tapestry
from errloom.utils import log

logger = log.getLogger(__name__)

# TODO "Proceesing" is ambiguous, not a true name

@dataclass
class GenerationConfig:
    """Configuration for batch generation behavior"""
    max_completion_length: int
    mask_truncated_completions: bool
    mask_env_responses: bool
    max_concurrent: int
    generation_timeout: float
    num_batches_ahead: int = 1

@dataclass
class ProcessingConfig:
    """Configuration for tokenization and processing"""
    tokenizer_class: PreTrainedTokenizerBase
    max_context_size: Optional[int] = None
    mask_env_responses: bool = False  # WTF is this? what is an "env response"?

@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    device: torch.device
    accelerator: Any
    process_index: int
    num_processes: int
    local_batch_size: int

@dataclass
class TapestryGravities:
    """Container for all reward-related scores"""
    # TODO this will go straight into @tapestry.py
    main_gravity: List[float]  # Primary reward scores (gravity)
    attractor_gravities: Dict[str, List[float]] = field(default_factory=dict)  # Individual attractor scores

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary format for backward compatibility"""
        result = {'reward': self.main_gravity}
        result.update(self.attractor_gravities)
        return result

    @classmethod
    def from_rollouts(cls, rollouts: List[Rollout]) -> 'TapestryGravities':
        """Create RewardScores from rollout data"""
        gravity_scores = []
        individual_rewards = {}

        for rollout in rollouts:
            gravity_scores.append(rollout.gravity)
            for key, value in rollout.gravities.items():
                if key not in individual_rewards:
                    individual_rewards[key] = []
                individual_rewards[key].append(value)

        return cls(
            main_gravity=gravity_scores,
            attractor_gravities=individual_rewards
        )

@dataclass
class TokenizedRollout:
    """Container for tokenized prompt and completion data"""
    # TODO this is wrong, the completion may be scattered across context. We are not doing simple Q/A context reinforcement. We need to unify these into just ids & mask
    prompt_ids: List[List[int]]
    prompt_mask: List[List[int]]
    completion_ids: List[List[int]]
    completion_mask: List[List[int]]
    # TODO it should contain a reference to the rollout and essentially be a transparent wrapper, just like @holophore.py

    def to_dict(self) -> Dict[str, List[List[int]]]:
        """Convert to dictionary format for backward compatibility"""
        return {
            "prompt_ids":      self.prompt_ids,
            "prompt_mask":     self.prompt_mask,
            "completion_ids":  self.completion_ids,
            "completion_mask": self.completion_mask,
        }

@dataclass
class BatchRequest:
    """Request for batch generation"""
    batch_id: int
    rows: Dataset
    generation_config: GenerationConfig
    processing_config: ProcessingConfig
    distributed_config: DistributedConfig

class BatchRequestBuilder:
    """Builder for creating BatchRequest instances with configuration objects"""

    def __init__(self):
        self._batch_id: Optional[int] = None
        self._rows: Optional[Dataset] = None
        self._generation_config: Optional[GenerationConfig] = None
        self._processing_config: Optional[ProcessingConfig] = None
        self._distributed_config: Optional[DistributedConfig] = None

    def with_batch_id(self, batch_id: int) -> 'BatchRequestBuilder':
        """Set the batch ID"""
        self._batch_id = batch_id
        return self

    def with_data(self, rows: Dataset) -> 'BatchRequestBuilder':
        """Set the dataset rows"""
        self._rows = rows
        return self

    def with_generation_config(self, config: GenerationConfig) -> 'BatchRequestBuilder':
        """Set the generation configuration"""
        self._generation_config = config
        return self

    def with_processing_config(self, config: ProcessingConfig) -> 'BatchRequestBuilder':
        """Set the processing configuration"""
        self._processing_config = config
        return self

    def with_distributed_config(self, config: DistributedConfig) -> 'BatchRequestBuilder':
        """Set the distributed training configuration"""
        self._distributed_config = config
        return self

    def build(self) -> BatchRequest:
        """Build the BatchRequest with all configurations"""
        if self._batch_id is None:
            raise ValueError("batch_id is required")
        if self._rows is None:
            raise ValueError("rows is required")
        if self._generation_config is None:
            raise ValueError("generation_config is required")
        if self._processing_config is None:
            raise ValueError("processing_config is required")
        if self._distributed_config is None:
            raise ValueError("distributed_config is required")

        return BatchRequest(
            batch_id=self._batch_id,
            rows=self._rows,
            generation_config=self._generation_config,
            processing_config=self._processing_config,
            distributed_config=self._distributed_config
        )

@dataclass
class BatchResult:
    """Result from batch generation"""
    batch_id: int
    processed_tokens: TokenizedRollout
    gravities: TapestryGravities
    generation_time: float = 0.0
    completions: List[Any] = field(default_factory=list)  # Store completions for logging
    prompts: List[Any] = field(default_factory=list)  # Store prompts for logging

    @property
    def processed_results(self) -> Dict[str, Any]:
        """Backward compatibility property for trainer access"""
        return self.processed_tokens.to_dict()

    @property
    def all_gravities_dict(self) -> Dict[str, List[float]]:
        """Backward compatibility property for trainer access"""
        return self.gravities.to_dict()


class AsyncBatchGenerator:
    """
    Manages asynchronous batch generation for GRPO training.

    This class runs generation in a separate thread, allowing training to continue
    while future batches are being generated. It maintains a queue of pending
    generation requests and completed results.
    """

    def __init__(
        self,
        loom: Loom,
        num_batches_ahead: int = 1,
        max_queue_size: Optional[int] = None,
        generation_timeout: float = 5 * 60,
        processing_strategy: Optional['TokenizationStrategy'] = None,
    ):
        self.loom = loom
        self.num_batches_ahead = num_batches_ahead
        self.max_queue_size = max_queue_size or max(num_batches_ahead * 2, 4)
        self.generation_timeout = generation_timeout

        # Default to chat format strategy
        self.pipe = ProcessingPipeline(strategy=processing_strategy or ChatFormatStrategy())

        # Queues for communication
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue(maxsize=self.num_batches_ahead)

        # Tracking
        self.pending_batches = set()  # batch_ids currently being processed
        self.completed_batches = {}  # batch_id -> BatchResult
        self.next_expected_batch = 0
        self.generation_times = deque(maxlen=100)  # Track recent generation times

        # Thread management
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.started = False

        # Synchronization
        self._lock = threading.Lock()

    def start(self):
        """Start the async generation worker thread"""
        if self.started:
            return

        logger.info("Starting ...")
        self.worker_thread = threading.Thread(
            target=self._generation_worker,
            daemon=True,
            name="AsyncBatchGenerator"
        )
        self.worker_thread.start()
        self.started = True

    def stop(self):
        """Stop the async generation worker thread"""
        if not self.started:
            return

        logger.info("Stopping ...")
        self.stop_event.set()

        # Send poison pill
        self.request_queue.put(None)
        if self.worker_thread:
            self.worker_thread.join(timeout=10.0)

        self.started = False

    def submit_batch(self, request: BatchRequest) -> bool:
        """
        Submit a batch for async generation.

        Returns:
            bool: True if submitted successfully, False if queue is full
        """
        if not self.started:
            raise RuntimeError("AsyncBatchGenerator not started")

        logger.debug(f"Submitting BatchRequest({request.batch_id})")
        with self._lock:
            if request.batch_id in self.pending_batches:
                return True  # Already submitted

            if len(self.pending_batches) >= self.num_batches_ahead:
                return False  # Queue full

            self.pending_batches.add(request.batch_id)

        self.request_queue.put(request)
        return True

    def get_batch(self, batch_id: int, timeout: Optional[float] = None) -> BatchResult:
        """
        Get a completed batch result. Blocks until the batch is ready.

        Args:
            batch_id: The batch ID to retrieve
            timeout: Maximum time to wait (uses generation_timeout if None)

        Returns:
            BatchResult: The completed batch result

        Raises:
            TimeoutError: If batch doesn't complete within timeout
            RuntimeError: If generation failed
        """
        timeout = timeout or self.generation_timeout
        start_time = time.time()

        logger.debug(f"get_batch({batch_id})")
        while True:
            # Check if already completed
            with self._lock:
                if batch_id in self.completed_batches:
                    result = self.completed_batches.pop(batch_id)
                    return result

            # Check for new results
            try:
                result = self.result_queue.get(timeout=0.1)
                with self._lock:
                    self.completed_batches[result.batch_id] = result
                    self.pending_batches.discard(result.batch_id)

                # If this is our batch, return it
                if result.batch_id == batch_id:
                    with self._lock:
                        result = self.completed_batches.pop(batch_id)
                        return result

            except queue.Empty:
                pass

            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Batch {batch_id} generation timed out after {timeout}s")

    def get_pending_count(self) -> int:
        """Get number of batches currently being generated"""
        with self._lock:
            return len(self.pending_batches)

    def get_completed_count(self) -> int:
        """Get number of completed batches waiting to be retrieved"""
        with self._lock:
            return len(self.completed_batches)

    def get_average_generation_time(self) -> float:
        """Get average generation time for recent batches"""
        if not self.generation_times:
            return 0.0
        return sum(self.generation_times) / len(self.generation_times)

    def should_submit_more(self) -> bool:
        """Check if we should submit more batches for generation"""
        with self._lock:
            total_pending = len(self.pending_batches) + len(self.completed_batches)
            return total_pending < self.num_batches_ahead

    def _generation_worker(self):
        """Worker thread that processes generation requests"""
        while not self.stop_event.is_set():
            try:
                # Get next request
                request = self.request_queue.get(timeout=0.1)
                if request is None:  # Poison pill
                    break

                # Generate batch
                start_time = time.time()
                result = self._generate_batch(request)
                generation_time = time.time() - start_time
                result.generation_time = generation_time
                self.generation_times.append(generation_time)

                # Put result in queue
                self.result_queue.put(result)

            except queue.Empty:
                continue

    def _generate_batch(self, request: BatchRequest) -> BatchResult:
        """
        Generate a single batch. This runs in the worker thread.
        """
        import time
        start_time = time.time()

        logger.push_info("Generate")

        # Convert rows dict to Dataset if needed
        rows = request.rows
        if isinstance(rows, dict):
            from datasets import Dataset
            rows = Dataset.from_dict(rows)
            logger.info(f"[cyan]🎯 Batch {request.batch_id}:[/] Processing {len(rows)} rows")

        # Call loom generation
        logger.info(f"[dim]⚡ Batch {request.batch_id}:[/] Starting loom rollouts...")
        loom_results = self.loom.weave(
            rows,
        )

        generation_time = time.time() - start_time
        logger.info(f"[green]✓ Batch {request.batch_id}:[/] Generated {len(loom_results.rollouts)} rollouts in {generation_time:.2f}s")

        logger.info(f"Loom generated {len(loom_results.rollouts)} rollouts")

        # Extract rewards from rollouts
        reward_scores = TapestryGravities.from_rollouts(loom_results.rollouts)

        # Extract completions and prompts
        completions = []
        prompts = []

        for rollout in loom_results.rollouts:
            # Use new method to get sample messages
            completions.append(rollout.get_sample_messages())
            # Extract prompt from rollout context
            if hasattr(rollout.context, 'messages') and rollout.context.messages:
                prompts.append(rollout.context.messages)
            elif hasattr(rollout.context, 'text'):
                prompts.append(rollout.context.text)
            else:
                # Fallback to row data
                prompts.append(rollout.row.get('prompt', ''))

        # Process results using the new pipeline
        processed_tokens = self.pipe.process_rollouts(
            loom_results,
            processing_config=request.processing_config,
            generation_config=request.generation_config
        )

        result = BatchResult(
            batch_id=request.batch_id,
            processed_tokens=processed_tokens,
            gravities=reward_scores,
            completions=completions,
            prompts=prompts
        )

        logger.pop()
        return result

# TODO there may be a better location for these, it's not exactly the batch generator's job either
# These functions used to be in Environment but that's definitely the wrong place, since
# the self parameter was unused anywhere in those function, indicating that the dependency on
# env is wrong and sets up a possible spaghettification in the near future if somebody
# were to actualize the binding with a self reference instead of introducing an arg ument.

def process_rollouts(
    rollouts: Tapestry,
    tokenizer_class: PreTrainedTokenizerBase,
    max_completion_length: int = -1,
    mask_truncated_completions: bool = False,
    mask_env_responses: bool = False,
) -> TokenizedRollout:
    """
    DEPRECATED: Legacy function for backward compatibility.
    Use ProcessingPipeline with ChatFormatStrategy instead.

    Returns:
        ProcessedTokens containing prompt_ids, prompt_mask, completion_ids, completion_mask
    """
    processing_config = ProcessingConfig(
        tokenizer_class=tokenizer_class,
        mask_env_responses=mask_env_responses
    )
    generation_config = GenerationConfig(
        max_completion_length=max_completion_length,
        mask_truncated_completions=mask_truncated_completions,
        mask_env_responses=mask_env_responses,
        max_concurrent=1,  # Dummy value for legacy compatibility
        generation_timeout=300.0,  # Dummy value for legacy compatibility
    )

    pipe = ProcessingPipeline(ChatFormatStrategy())
    return pipe.process_rollouts(rollouts, processing_config, generation_config)

def process_completion(
    rollout: Rollout,
    processing_class: PreTrainedTokenizerBase
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    DEPRECATED: This function is a Verifiers legacy that assumes simple prompt->completion format.
    Errloom uses dynamic context scaffolding with chat format exclusively.
    This function should never be called in the current architecture.
    """
    raise NotImplementedError(
        "process_completion is deprecated in Errloom. "
        "All processing should use process_chat_format with dynamic context scaffolding."
    )

def process_chat_format(
    rollout: Rollout,
    tokenizer: PreTrainedTokenizerBase,
    mask_env_responses: bool = False
) -> TokenizedRollout:
    """
    Process chat format conversations using incremental prefixes.

    Logic:
    1. For each step, tokenize conversation prefix (prompt + completion[:i])
    2. Calculate token differences between steps to get individual message tokens
    3. Apply masking for intermediate responses if needed

    Returns:
        TokenizedRollout containing prompt_ids, prompt_mask, completion_ids, completion_mask
    """

    # Extract prompt messages from rollout context
    messages = rollout.context.messages
    if not isinstance(messages, list):
        messages = list(messages)

    # Get completion messages from samples - they're automatically maintained by rollout
    completion = rollout.get_sample_messages()

    # tokenize just the prompt
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert isinstance(prompt_text, str)
    prompt_ids = tokenizer.encode(prompt_text)
    prompt_mask = [1] * len(prompt_ids)

    # track completion tokens and masks by processing incrementally
    completion_ids = []
    completion_mask = []

    # previous tokenization (starts with just prompt)
    prev_ids = prompt_ids

    # process each completion message incrementally
    for i, msg in enumerate(completion):
        # create conversation prefix: prompt + completion[:i+1]
        # Type cast to resolve linter - both are message dicts in practice
        conversation_prefix = list(messages) + completion[:i + 1]

        # tokenize the full prefix
        prefix_text = tokenizer.apply_chat_template(conversation_prefix, tokenize=False, add_generation_prompt=False, )
        assert isinstance(prefix_text, str), f"Expected string from apply_chat_template, got {type(prefix_text)}"

        current_ids = tokenizer.encode(prefix_text)
        assert current_ids[:len(prev_ids) - 1] == prev_ids[:-1], f"Tokenization difference in chat format. Current ids: {current_ids[:len(prev_ids) - 1]}, previous ids: {prev_ids[:-1]}"

        # add new tokens to completion tokens
        new_tokens = current_ids[len(prev_ids):]
        assert len(new_tokens) > 0, f"No new tokens in chat format. Current ids: {current_ids}, previous ids: {prev_ids}"
        completion_ids.extend(new_tokens)

        # create mask
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            msg_mask = [1] * len(new_tokens)
        elif isinstance(msg, dict) and msg.get("role") != "assistant" and mask_env_responses:
            # mask intermediate 'user' and/or 'tool' messages
            msg_mask = [0] * len(new_tokens)
        else:
            # default to not masking
            msg_mask = [1] * len(new_tokens)

        completion_mask.extend(msg_mask)
        # update previous tokenization for next iteration
        prev_ids = current_ids
        assert len(completion_ids) == len(completion_mask), f"Length mismatch in chat format. \
Completion ids: {completion_ids}, completion mask: {completion_mask}. \
This often occurs with models whose tokenizer chat templates discard <think> tokens \
from previous turns, such as Qwen3 or DeepSeek-R1-Distill models. \
For Qwen3 models, you may want to replace the chat template with the Qwen2.5 chat template. \
Model copies with swapped templates are available here: https://huggingface.co/collections/willcb/qwen3-68434f4883925bfdb4570ee5"

    return TokenizedRollout(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask
    )

class TokenizationStrategy(ABC):
    """Abstract base class for tokenization strategies"""

    @abstractmethod
    def process_rollout(self, rollout: Rollout, config: ProcessingConfig) -> TokenizedRollout:
        """Process a single rollout into tokenized format"""
        pass

class ChatFormatStrategy(TokenizationStrategy):
    """Strategy for processing chat format conversations"""

    def process_rollout(self, rollout: Rollout, config: ProcessingConfig) -> TokenizedRollout:
        """Process chat format using incremental prefixes"""
        # Extract prompt messages from rollout context
        messages = rollout.context.messages
        if not isinstance(messages, list):
            messages = list(messages)

        # Get completion messages from samples
        completion = rollout.get_sample_messages()

        # Tokenize just the prompt
        prompt_text = config.tokenizer_class.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(prompt_text, str)
        prompt_ids = config.tokenizer_class.encode(prompt_text)
        prompt_mask = [1] * len(prompt_ids)

        # Track completion tokens and masks by processing incrementally
        completion_ids = []
        completion_mask = []
        prev_ids = prompt_ids

        # Process each completion message incrementally
        for i, msg in enumerate(completion):
            # Create conversation prefix: prompt + completion[:i+1]
            conversation_prefix = list(messages) + completion[:i + 1]

            # Tokenize the full prefix
            prefix_text = config.tokenizer_class.apply_chat_template(
                conversation_prefix,
                tokenize=False,
                add_generation_prompt=False,
            )
            assert isinstance(prefix_text, str), f"Expected string from apply_chat_template, got {type(prefix_text)}"
            current_ids = config.tokenizer_class.encode(prefix_text)
            assert current_ids[:len(prev_ids) - 1] == prev_ids[:-1], f"Tokenization difference in chat format"

            # Add new tokens to completion tokens
            new_tokens = current_ids[len(prev_ids):]
            assert len(new_tokens) > 0, f"No new tokens in chat format"
            completion_ids.extend(new_tokens)

            # Create mask based on message role
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                msg_mask = [1] * len(new_tokens)
            elif isinstance(msg, dict) and msg.get("role") != "assistant" and config.mask_env_responses:
                # Mask intermediate 'user' and/or 'tool' messages
                msg_mask = [0] * len(new_tokens)
            else:
                # Default to not masking
                msg_mask = [1] * len(new_tokens)

            completion_mask.extend(msg_mask)
            prev_ids = current_ids

        assert len(completion_ids) == len(completion_mask), f"Length mismatch in chat format"

        return TokenizedRollout(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask
        )

class ProcessingPipeline:
    """Orchestrates the tokenization process using strategies"""

    # TODO this is overengineered

    def __init__(self, strategy: TokenizationStrategy):
        self.strategy = strategy

    def process_rollouts(self,
                         tapestry: Tapestry,
                         processing_config: ProcessingConfig,
                         generation_config: GenerationConfig) -> TokenizedRollout:
        """Process all rollouts using the configured strategy"""
        # TODO this is wrong, the completion may be scattered across context. We are not doing simple Q/A context reinforcement. We need to unify these
        all_prompt_ids = []
        all_prompt_masks = []
        all_completion_ids = []
        all_completion_masks = []

        for rollout in tapestry:
            tokenized = self.strategy.process_rollout(rollout, processing_config)

            # Apply truncation if needed
            completion_ids = tokenized.completion_ids
            completion_mask = tokenized.completion_mask

            if generation_config.mask_truncated_completions and \
                0 < generation_config.max_completion_length < len(completion_ids):
                completion_ids = completion_ids[:generation_config.max_completion_length]
                completion_mask = [0] * len(completion_ids)

            all_prompt_ids.append(tokenized.prompt_ids)
            all_prompt_masks.append(tokenized.prompt_mask)
            all_completion_ids.append(completion_ids)
            all_completion_masks.append(completion_mask)

        return TokenizedRollout(
            prompt_ids=all_prompt_ids,
            prompt_mask=all_prompt_masks,
            completion_ids=all_completion_ids,
            completion_mask=all_completion_masks,
        )
