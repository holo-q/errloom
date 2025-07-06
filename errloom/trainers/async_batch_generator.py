import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset
from openai import OpenAI
from transformers import PreTrainedTokenizerBase

from errloom import Loom
from errloom.states import Rollout, Rollouts

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """Request for batch generation"""
    batch_id: int
    rows: Dataset
    processing_class: Any
    mask_env_responses: bool
    max_completion_length: int
    mask_truncated_completions: bool
    max_concurrent: int
    device: torch.device
    accelerator: Any
    process_index: int
    num_processes: int
    local_batch_size: int


@dataclass
class BatchResult:
    """Result from batch generation"""
    batch_id: int
    processed_results: Dict[str, Any]
    generation_time: float = 0.0
    all_reward_dict: Dict[str, List[float]] = field(default_factory=dict)  # All reward scores
    completions: List[Any] = field(default_factory=list)  # Store completions for logging
    prompts: List[Any] = field(default_factory=list)  # Store prompts for logging


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
        model_name: str,
        client: Optional[OpenAI] = None,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        port: int = 8000,
        max_concurrent_requests: int = 100,
        generation_timeout: float = 300.0,  # 5 minutes default
    ):
        self.loom = loom
        self.client = client
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.port = port
        self.max_concurrent_requests = max_concurrent_requests
        self.generation_timeout = generation_timeout

        # Queues for communication
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue(maxsize=self.max_concurrent_requests)

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

        with self._lock:
            if request.batch_id in self.pending_batches:
                return True  # Already submitted

            if len(self.pending_batches) >= self.max_concurrent_requests:
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

        while True:
            # Check if already completed
            with self._lock:
                if batch_id in self.completed_batches:
                    return self.completed_batches.pop(batch_id)

            # Check for new results
            try:
                result = self.result_queue.get(timeout=0.1)
                with self._lock:
                    self.completed_batches[result.batch_id] = result
                    self.pending_batches.discard(result.batch_id)

                # If this is our batch, return it
                if result.batch_id == batch_id:
                    with self._lock:
                        return self.completed_batches.pop(batch_id)

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
            return total_pending < self.max_concurrent_requests

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

    def _generate_batch(self, request: BatchRequest, all_reward_dict=None) -> BatchResult:
        """
        Generate a single batch. This runs in the worker thread.
        """
        # Call environment generation
        loom_results = self.loom.unroll(
            request.rows,
            # model=self.model_name,
            # base_url=self.base_url,
            # api_key=self.api_key,
            # port=self.port,
            # max_concurrent=request.max_concurrent,
        )

        logger.info(f"Loom generated {len(loom_results.rollouts)} rollouts")
        # Send rollouts to task queue

        # Extract all reward-related keys
        # all_reward_dict = {}
        # reward_keys = [k for k in loom_results.keys() if k.endswith('_func') or k == 'reward']
        # for key in reward_keys:
        #     all_reward_dict[key] = loom_results[key]

        # Process results
        processed_results = process_rollouts(
            loom_results,
            processing_class=request.processing_class,
            mask_env_responses=request.mask_env_responses,
            max_completion_length=request.max_completion_length,
            mask_truncated_completions=request.mask_truncated_completions
        )

        return BatchResult(
            batch_id=request.batch_id,
            processed_results=processed_results,
            all_reward_dict=all_reward_dict,
            completions=loom_results['completion'],
            prompts=loom_results['prompt']
        )

# TODO there may be a better location for these, it's not exactly the batch generator's job either
# These functions used to be in Environment but that's definitely the wrong place, since
# the self parameter was unused anywhere in those function, indicating that the dependency on
# env is wrong and sets up a possible spaghettification in the near future if somebody
# were to actualize the binding with a self reference instead of introducing an arg ument.


def process_rollouts(
    rollouts: Rollouts,
    processing_class: PreTrainedTokenizerBase,
    max_completion_length: int = -1,
    mask_truncated_completions: bool = False,
    mask_env_responses: bool = False,
) -> Dict[str, List[Any]]: # TODO dataclass
    """
    Main tokenization pipeline that handles both chat and completion formats.

    Returns:
        Dict with prompt_ids, prompt_mask, completion_ids, completion_mask, rewards
    """
    # TODO: states + rewards for intermediate-reward objectives

    # Determine format from first prompt
    is_chat_format = isinstance(rollouts[0].context, list)

    all_prompt_ids = []
    all_prompt_masks = []
    all_completion_ids = []
    all_completion_masks = []

    for rollout in rollouts:
        # Format-specific processing
        if is_chat_format:
            prompt_ids, prompt_mask, completion_ids, completion_mask = process_chat_format(
                rollout, processing_class, mask_env_responses
            )
        else:
            prompt_ids, prompt_mask, completion_ids, completion_mask = process_completion(
                rollout, processing_class
            )
        if mask_truncated_completions and max_completion_length > 0 and len(completion_ids) > max_completion_length:
            completion_ids = completion_ids[:max_completion_length]
            completion_mask = [0] * len(completion_ids)
        all_prompt_ids.append(prompt_ids)
        all_prompt_masks.append(prompt_mask)
        all_completion_ids.append(completion_ids)
        all_completion_masks.append(completion_mask)

    return {
        "prompt_ids":      all_prompt_ids,
        "prompt_mask":     all_prompt_masks,
        "completion_ids":  all_completion_ids,
        "completion_mask": all_completion_masks,
    }

def process_completion(
    rollout: Rollout,
    processing_class: PreTrainedTokenizerBase
) -> Tuple[List[int], List[int], List[int], List[int]]: # TODO dataclass
    """
    Process completion format text.

    Logic:
    1. Tokenize prompt separately to get boundary
    2. Tokenize completion
    3. Create masks (prompt mask all 1s, completion mask handles EOS)

    Returns:
        prompt_ids, prompt_mask, completion_ids, completion_mask
    """
    # Tokenize prompt
    prompt_ids = processing_class.encode(prompt)
    prompt_mask = [1] * len(prompt_ids)

    # Tokenize completion
    completion_ids = processing_class.encode(completion)
    completion_mask = [1] * len(completion_ids)

    return prompt_ids, prompt_mask, completion_ids, completion_mask

def process_chat_format(
    rollout: Rollout,
    processing_class: PreTrainedTokenizerBase,
    mask_env_responses: bool = False
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Process chat format conversations using incremental prefixes.

    Logic:
    1. For each step, tokenize conversation prefix (prompt + completion[:i])
    2. Calculate token differences between steps to get individual message tokens
    3. Apply masking for intermediate responses if needed

    Returns:
        prompt_ids, prompt_mask, completion_ids, completion_mask
    """
    prompt = rollout.context

    # tokenize just the prompt
    prompt_text = processing_class.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    assert isinstance(prompt_text, str)
    prompt_ids = processing_class.encode(prompt_text)
    prompt_mask = [1] * len(prompt_ids)

    # track completion tokens and masks by processing incrementally
    completion_ids = []
    completion_mask = []

    # previous tokenization (starts with just prompt)
    prev_ids = prompt_ids

    # process each completion message incrementally
    for i, msg in enumerate(completion):
        # create conversation prefix: prompt + completion[:i+1]
        conversation_prefix = prompt + completion[:i + 1]

        # tokenize the full prefix
        prefix_text = processing_class.apply_chat_template(
            conversation_prefix,
            tokenize=False,
            add_generation_prompt=False,
        )
        assert isinstance(prefix_text, str), f"Expected string from apply_chat_template, got {type(prefix_text)}"
        current_ids = processing_class.encode(prefix_text)
        assert current_ids[:len(prev_ids) - 1] == prev_ids[:-1], f"Tokenization difference in chat format. Current ids: {current_ids[:len(prev_ids) - 1]}, previous ids: {prev_ids[:-1]}"

        # add new tokens to completion tokens
        new_tokens = current_ids[len(prev_ids):]
        assert len(new_tokens) > 0, f"No new tokens in chat format. Current ids: {current_ids}, previous ids: {prev_ids}"
        completion_ids.extend(new_tokens)

        # create mask
        if msg["role"] == "assistant":
            msg_mask = [1] * len(new_tokens)
        elif msg["role"] != "assistant" and mask_env_responses:
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

    return prompt_ids, prompt_mask, completion_ids, completion_mask
