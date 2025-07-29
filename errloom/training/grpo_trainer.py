# adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py

import logging
import queue
import threading
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sized, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import datasets
import numpy as np
import openai
import torch
import wandb
from accelerate.utils import broadcast_object_list, gather_object, is_peft_model
from peft import get_peft_model, PeftConfig
from torch.utils.data import DataLoader, Sampler
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import seed_worker
from trl.models import create_reference_model, prepare_deepspeed
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.utils import (
    disable_dropout_in_model,
    pad,
    selective_log_softmax
)

from errloom.loom import Loom
from errloom.training.async_batch_generator import (
    AsyncBatchGenerator, 
    BatchRequest, 
    GenerationConfig, 
    ProcessingConfig, 
    DistributedConfig
)
from errloom.training.async_dataloader_wrapper import AsyncDataLoaderWrapper
from errloom.training.grpo_config import GRPOConfig
from errloom.utils.log import LogContext
from errloom.utils import log
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

import torch._dynamo
torch._dynamo.config.suppress_errors = True  # type: ignore

@dataclass
class RLLossResult:
    """Result from RL algorithm loss computation"""
    loss: torch.Tensor
    metrics: Dict[str, float]
    clipping_stats: Dict[str, torch.Tensor]

@dataclass
class PolicyMetrics:
    """Container for policy-related metrics during training"""
    importance_ratios: torch.Tensor
    advantages: torch.Tensor
    clipped_ratios: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: Optional[torch.Tensor] = None

class RLAlgorithm(ABC):
    """Abstract base class for RL algorithms (GRPO, GSPO, etc.)"""
    
    def __init__(self, config: Any, logger: Any):
        self.config = config
        self.logger = logger
    
    @abstractmethod
    def compute_importance_ratios(self, 
                                 per_token_logps: torch.Tensor,
                                 old_per_token_logps: torch.Tensor,
                                 completion_mask: torch.Tensor) -> torch.Tensor:
        """Compute importance ratios for policy updates"""
        pass
    
    @abstractmethod
    def compute_advantages(self, 
                          rewards: torch.Tensor,
                          num_generations: int) -> torch.Tensor:
        """Compute advantages from rewards"""
        pass
    
    @abstractmethod
    def compute_policy_loss(self,
                           importance_ratios: torch.Tensor,
                           advantages: torch.Tensor,
                           completion_mask: torch.Tensor) -> RLLossResult:
        """Compute the policy loss with algorithm-specific logic"""
        pass
    
    def get_algorithm_name(self) -> str:
        """Return the algorithm name for logging"""
        return self.__class__.__name__.replace('Algorithm', '').upper()

class GRPOAlgorithm(RLAlgorithm):
    """GRPO (Group Relative Policy Optimization) implementation"""
    
    def compute_importance_ratios(self, 
                                 per_token_logps: torch.Tensor,
                                 old_per_token_logps: torch.Tensor,
                                 completion_mask: torch.Tensor) -> torch.Tensor:
        """GRPO uses token-level importance ratios"""
        return torch.exp(per_token_logps - old_per_token_logps)
    
    def compute_advantages(self, 
                          rewards: torch.Tensor,
                          num_generations: int) -> torch.Tensor:
        """GRPO advantage computation with group normalization"""
        # Always use full batch statistics
        mean_grouped = rewards.view(-1, num_generations).mean(dim=1)
        std_grouped = rewards.view(-1, num_generations).std(dim=1)

        # Normalize the rewards to compute advantages
        mean_grouped = mean_grouped.repeat_interleave(num_generations, dim=0)
        std_grouped = std_grouped.repeat_interleave(num_generations, dim=0)
        advantages = rewards - mean_grouped

        if self.config.scale_rewards:
            advantages = advantages / (std_grouped + 1e-4)

        return advantages
    
    def compute_policy_loss(self,
                           importance_ratios: torch.Tensor,
                           advantages: torch.Tensor,
                           completion_mask: torch.Tensor) -> RLLossResult:
        """GRPO policy loss with token-level clipping"""
        # GRPO clipping parameters
        epsilon_low = self.config.epsilon
        epsilon_high = getattr(self.config, 'epsilon_high', self.config.epsilon)
        
        # Token-level clipping
        coef_1 = importance_ratios
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

        if hasattr(self.config, 'delta') and self.config.delta is not None:
            # Use clamp instead of min to handle tensor-float comparison
            per_token_loss1 = torch.clamp(coef_1, max=self.config.delta) * advantages.unsqueeze(1)
        else:
            # Original GRPO clipping (only lower bound implicitly applied by the final min)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)

        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        # Apply loss aggregation based on loss_type
        if self.config.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.config.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.config.loss_type == "dr_grpo":
            max_completion_length = getattr(self.config, 'max_completion_length', completion_mask.size(-1))
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Compute clipping statistics
        is_low_clipped = (coef_1 < 1 - epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped
        
        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()
        
        metrics = {}
        clipping_stats = {
            'low_clip': low_clip,
            'high_clip': high_clip,
            'clip_ratio': clip_ratio
        }
        
        return RLLossResult(
            loss=loss,
            metrics=metrics,
            clipping_stats=clipping_stats
        )

class GSPOAlgorithm(RLAlgorithm):
    """GSPO (Group Sequence Policy Optimization) implementation"""
    
    def compute_importance_ratios(self, 
                                 per_token_logps: torch.Tensor,
                                 old_per_token_logps: torch.Tensor,
                                 completion_mask: torch.Tensor) -> torch.Tensor:
        """GSPO uses sequence-level importance ratios with length normalization"""
        # Compute log ratio for each token
        log_ratios = per_token_logps - old_per_token_logps
        
        # Sum over sequence length, weighted by completion mask
        sequence_log_ratios = (log_ratios * completion_mask).sum(dim=1)
        sequence_lengths = completion_mask.sum(dim=1)
        
        # Length-normalized sequence importance ratio: (π_θ(y|x) / π_θ_old(y|x))^(1/|y|)
        length_normalized_log_ratios = sequence_log_ratios / sequence_lengths
        sequence_importance_ratios = torch.exp(length_normalized_log_ratios)
        
        # Expand to match per-token shape for consistent interface
        return sequence_importance_ratios.unsqueeze(1).expand_as(completion_mask)
    
    def compute_advantages(self, 
                          rewards: torch.Tensor,
                          num_generations: int) -> torch.Tensor:
        """GSPO uses same group-based advantage computation as GRPO"""
        # Always use full batch statistics
        mean_grouped = rewards.view(-1, num_generations).mean(dim=1)
        std_grouped = rewards.view(-1, num_generations).std(dim=1)

        # Normalize the rewards to compute advantages
        mean_grouped = mean_grouped.repeat_interleave(num_generations, dim=0)
        std_grouped = std_grouped.repeat_interleave(num_generations, dim=0)
        advantages = rewards - mean_grouped

        if self.config.scale_rewards:
            advantages = advantages / (std_grouped + 1e-4)

        return advantages
    
    def compute_policy_loss(self,
                           importance_ratios: torch.Tensor,
                           advantages: torch.Tensor,
                           completion_mask: torch.Tensor) -> RLLossResult:
        """GSPO policy loss with sequence-level clipping"""
        # GSPO uses different clipping ranges (typically much smaller)
        epsilon = getattr(self.config, 'gspo_epsilon', 0.05)  # GSPO typically uses smaller epsilon
        
        # Extract sequence-level importance ratios (they're repeated across tokens)
        seq_importance_ratios = importance_ratios[:, 0]  # First token has the sequence ratio
        
        # Sequence-level clipping
        clipped_ratios = torch.clamp(seq_importance_ratios, 1 - epsilon, 1 + epsilon)
        
        # Compute sequence-level losses
        loss1 = seq_importance_ratios * advantages
        loss2 = clipped_ratios * advantages
        sequence_loss = -torch.min(loss1, loss2)
        
        # Average over batch
        loss = sequence_loss.mean()
        
        # Compute clipping statistics
        is_clipped = (seq_importance_ratios < 1 - epsilon) | (seq_importance_ratios > 1 + epsilon)
        clip_ratio = is_clipped.float().mean()
        
        metrics = {}
        clipping_stats = {
            'clip_ratio': clip_ratio,
            'mean_importance_ratio': seq_importance_ratios.mean(),
            'std_importance_ratio': seq_importance_ratios.std()
        }
        
        return RLLossResult(
            loss=loss,
            metrics=metrics,
            clipping_stats=clipping_stats
        )

@dataclass
class BatchData:
    """Container for gathered batch data from all processes"""
    prompts: List[Any]
    answers: List[Any] 
    tasks: List[Any]
    infos: List[Any]

@dataclass
class OptimizerConfig:
    """Container for optimizer and scheduler configuration"""
    optimizer: Optional[torch.optim.Optimizer] = None
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    
    def to_tuple(self) -> tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]]:
        """Convert to tuple format for backward compatibility with transformers"""
        return (self.optimizer, self.lr_scheduler)

class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i: i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = torch.arange(12).reshape(6, 2)
        >>> y = torch.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size: (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]

def shuffle_tensor_dict(tensor_dict: dict[str, Optional[torch.Tensor]]) -> dict[str, Optional[torch.Tensor]]:
    """
    Shuffles a dictionary of tensors along the first dimension in unison.

    Example:
        >>> x = torch.arange(6).reshape(3, 2)
        >>> y = torch.arange(3).reshape(3, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> shuffle_tensor_dict(tensor_dict)
        {'x': tensor([[2, 3],
                      [0, 1],
                      [4, 5]]),
         'y': tensor([[1],
                      [0],
                      [2]])}
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    batch_size = first_tensor.shape[0]
    permutation = torch.randperm(batch_size)
    return {key: tensor[permutation] if tensor is not None else None for key, tensor in tensor_dict.items()}

def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])

def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class GRPOTrainer(Trainer):
    def __init__(
        self,
        loom: Loom,
        model: PreTrainedModel,
        args: GRPOConfig,
        tokenizer: PreTrainedTokenizerBase,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: Optional[OptimizerConfig] = None,
        peft_config: Optional[PeftConfig] = None,
        algorithm: Optional[RLAlgorithm] = None
    ):
        # Initialize the logger with proper hierarchy
        self.logger = log.getLogger(f'errloom.training.{self.__class__.__name__}')
        
        # Initialize RL algorithm (default to GRPO for backward compatibility)
        if algorithm is None:
            algorithm = GRPOAlgorithm(config=args, logger=self.logger)
        self.algorithm = algorithm
        
        # Initialize modular training pipeline
        self.training_pipeline = TrainingPipelineFactory.create_default(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_iterations=args.num_accum
        )
        
        algorithm_name = self.algorithm.get_algorithm_name()
        with LogContext(f"🔧 Configuring {algorithm_name} Trainer...", "trainer_config", logger=self.logger):
            self.logger.info(f"[cyan]Training Configuration:[/]")
            self.logger.info(f"  • Learning rate: [green]{args.learning_rate}[/]")
            self.logger.info(f"  • Batch size: [green]{args.per_device_train_batch_size}[/]")
            self.logger.info(f"  • Max steps: [green]{args.max_steps}[/]") 
            self.logger.info(f"  • Beta (KL coeff): [green]{args.beta}[/]")
            self.logger.info(f"  • Epsilon (clipping): [green]{args.epsilon}[/]")
            if hasattr(args, 'disable_nccl_init') and args.disable_nccl_init:
                self.logger.info(f"  • [yellow]NCCL disabled for local testing[/]")

        self.loom: Loom | None = loom

        # Models
        if peft_config is not None:
            model = get_peft_model(model, peft_config)  # type: ignore
            # Override sync_ref_model if PEFT is used since ref_model will be None
            if args.sync_ref_model:
                self.logger.warning("sync_ref_model=True is not compatible with PEFT. Setting sync_ref_model=False.")
                args.sync_ref_model = False

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)  # type: ignore

        # Suppress irrelevant warning
        model.warnings_issued["estimate_tokens"] = True

        # Tokenizer pad token
        if tokenizer.pad_token is None:  # type: ignore
            tokenizer.pad_token = tokenizer.eos_token  # type: ignore

        # Training arguments
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.max_prompt_length = args.max_context_size
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_rows  # = G in the GRPO paper
        self.max_concurrent = args.max_concurrent
        self.max_num_processes = args.max_num_processes
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.presence_penalty = args.presence_penalty
        self.frequency_penalty = args.frequency_penalty
        self.top_k = args.top_k
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions
        self.delta = args.delta

        # Reference model parameters
        self.beta = args.beta
        self.sync_ref_model = args.sync_ref_model
        self.generation_batch_size: int = args.batch_size  # type: ignore

        # Multi-step
        self.num_iterations = args.num_accum  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self._step = 0
        self._buffered_inputs: Optional[List[Dict[str, Optional[torch.Tensor]]]] = None

        # Data
        self.shuffle_dataset = args.shuffle_dataset
        train_dataset = loom.get_train_data()
        assert train_dataset is not None

        eval_dataset = loom.get_bench_data()

        # Dataset filtering removed - context length management handled by loom pipeline

        # dummy data collator
        def data_collator(features):
            return features

        self.tokenizer = tokenizer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
            optimizers=optimizers.to_tuple() if optimizers else (None, None),
        )

        # Reference model
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            model_id = model.config._name_or_path
            model_init_kwargs = {"torch_dtype": "auto"}
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)  # type: ignore

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Initialize structured metrics
        self._metrics = ModeMetrics()
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print

        # Environment integration parameters
        self.mask_env_responses = args.mask_env_responses
        self.max_concurrent = args.max_concurrent

        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._textual_logs = TextualLogs(maxlen=maxlen)

        # OpenAI client for Environment generation (using vLLM server)
        host = args.vllm_server_host
        port = args.vllm_server_port
        vllm_base_url = f"http://{host}:{port}/v1"
        import httpx
        self.oai_client = openai.OpenAI(
            base_url=vllm_base_url,
            api_key="EMPTY",
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=args.max_concurrent),
                timeout=args.async_generation_timeout)
        )

        # vLLM client for weight syncing only; only import if used
        from errloom.interop.vllm_client import VLLMClient
        self.vllm_client = VLLMClient(
            host=host,
            port=port,
            connection_timeout=args.vllm_server_timeout
        )
        # Only initialize communicator on the main process
        # Other processes will only use the client for non-NCCL operations
        if self.accelerator.is_main_process and not getattr(args, 'disable_nccl_init', False):
            with LogContext("🔗 Initializing vLLM Communicator...", "vllm_comm", logger=self.logger):
                self.logger.info(f"[cyan]Connecting to vLLM server at {host}:{port}[/]")
                self.vllm_client.init_communicator()
                self.logger.info(f"[green]✓[/] vLLM communicator ready")
        else:
            if getattr(args, 'disable_nccl_init', False):
                self.logger.info(f"[yellow]⚠[/] NCCL communicator disabled for local testing")
            else:
                self.logger.info(f"[dim]Non-main process, skipping NCCL init[/]")

        self._last_loaded_step = 0  # Initialize to 0 since vLLM already has initial weights
        self.model_accepts_loss_kwargs = False
        # Weight updates to vLLM happen only when generating new completions
        # Frequency: every (gradient_accumulation_steps * num_iterations) training steps
        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        self.accelerator.wait_for_everyone()

        # Reference model
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        if self.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))  # type: ignore


        # # Async generation setup
        # self.batch_generator = AsyncBatchGenerator(
        #     loom=loom,
        #     model_name=args.model_name,
        #     client=self.oai_client,
        #     base_url=vllm_base_url,
        #     api_key="EMPTY",
        #     port=port,
        #     max_concurrent_requests=args.max_concurrent,
        #     generation_timeout=args.async_generation_timeout,
        # )
        # Async generation setup
        self._next_batch_id: int = 0
        self._async_started = False
        self.num_batches_ahead = args.num_batches_ahead

        # num_batches_ahead=0 will behave synchronously (submit and wait immediately)
        self.async_generator = AsyncBatchGenerator(
            loom=self.loom,
            # client=self.oai_client,
            # model_name=self._get_model_name(),
            generation_timeout=args.async_generation_timeout,
        )

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        batch_size = self._train_batch_size * self.gradient_accumulation_steps  # type: ignore

        dataloader_params = {
            "batch_size":         batch_size,  # type: ignore (None case handled by config __post_init__)
            "collate_fn":         data_collator,
            "num_workers":        self.args.dataloader_num_workers,
            "pin_memory":         self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dataloader = DataLoader(train_dataset, **dataloader_params)  # type: ignore

        # Always wrap with AsyncDataLoaderWrapper for consistent behavior
        # Store the wrapped dataloader for async access
        self._async_dataloader = AsyncDataLoaderWrapper(
            dataloader,
            buffer_size=max(5, self.num_batches_ahead * 2)
        )
        return self.accelerator.prepare(self._async_dataloader)

    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |    Accum step 0     |
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first gradient_accumulation_steps (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  grad_accum=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second gradient_accumulation_steps (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...

        return RepeatSampler(
            data_source=self.train_dataset,  # type: ignore
            mini_repeat_count=self.num_generations,
            batch_size=self.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.gradient_accumulation_steps,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()  # type: ignore
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        assert isinstance(gradient_checkpointing_kwargs, dict)
        use_reentrant = gradient_checkpointing_kwargs.get("use_reentrant", True)

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _inner_training_loop(self, *args, **kwargs):
        """Override to ensure async generator is stopped when training ends"""
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            # Clean up async generator on all processes
            if self.async_generator and self._async_started and self.accelerator.is_main_process:
                self.async_generator.stop()
            self._async_started = False

    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep=None):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model
        last_hidden_state = unwrapped_model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i: i + batch_size]
            attention_mask_batch = attention_mask[i: i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed
            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        # Ensure all processes are synchronized before weight update
        self.accelerator.wait_for_everyone()

        # Debug log
        self.logger.info(f"Process {self.accelerator.process_index}: Starting weight sync to vLLM")

        # ALL processes must participate in model operations for DeepSpeed ZeRO-3
        if is_peft_model(self.model):
            # With PEFT and DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging
            with gather_if_zero3(list(self.model.parameters())):  # type: ignore
                self.model.merge_adapter()  # type: ignore

                # Update vLLM weights while parameters are gathered
                for name, param in self.model.named_parameters():  # type: ignore
                    # When using PEFT, we need to recover the original parameter name and discard some parameters
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if self.model.prefix in name:  # type: ignore
                        continue
                    # When module to save, remove its prefix and discard the original module
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")

                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

                self.model.unmerge_adapter()  # type: ignore
        else:
            # For non-PEFT models, gather and update each parameter individually
            for name, param in self.model.named_parameters():  # type: ignore
                with gather_if_zero3([param]):
                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

        # Reset cache on vLLM (main process only)
        if self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()

        # Ensure all processes wait for the main process to finish updating weights
        self.accelerator.wait_for_everyone()

    def _get_sampling_args(self) -> Dict[str, Any]:
        """Get sampling arguments for Environment generation."""
        args = {
            'temperature':       self.temperature,
            'top_p':             self.top_p,
            'max_tokens':        self.max_completion_length,
            'n':                 1,
            'presence_penalty':  self.presence_penalty,
            'frequency_penalty': self.frequency_penalty,
            'extra_body':        {
                'top_k':              self.top_k,
                'min_p':              self.min_p,
                'repetition_penalty': self.repetition_penalty,
            }
        }
        return args

    def _get_model_name(self) -> str:
        """Get model name for Environment generation."""
        return self.model.config._name_or_path  # type: ignore

    def _ids_to_tensors(self,
                        prompt_ids: List[List[int]],
                        prompt_mask: List[List[int]],
                        completion_ids: List[List[int]],
                        completion_mask: List[List[int]],
                        device: torch.device) -> Dict[str, torch.Tensor]:
        ids = [prompt_ids[i] + completion_ids[i] for i in range(len(prompt_ids))]
        mask = [prompt_mask[i] + completion_mask[i] for i in range(len(prompt_mask))]
        max_len = max(len(ids[i]) for i in range(len(ids)))
        ids = [torch.cat([
            torch.tensor(ids[i], dtype=torch.long, device=device),
            torch.zeros(max_len - len(ids[i]), dtype=torch.long, device=device)
        ]) for i in range(len(ids))]
        mask = [torch.cat([
            torch.tensor(mask[i], dtype=torch.long, device=device),
            torch.zeros(max_len - len(mask[i]), dtype=torch.long, device=device)
        ]) for i in range(len(mask))]
        ids = torch.stack(ids, dim=0)
        mask = torch.stack(mask, dim=0)
        return {
            'ids':  ids,
            'mask': mask
        }

    def _gather_batch_data(self, batch_offset: int = 0) -> BatchData:
        """Public interface for batch data gathering"""
        return self.training_pipeline.batch_data_strategy.gather_batch_data(self, batch_offset)
    
    def _gather_batch_data_impl(self, batch_offset: int = 0) -> BatchData:
        """
        Gather batch data from all processes.

        Args:
            batch_offset: 0 for current batch, >0 for future batches (peek ahead)

        Returns:
            BatchData containing prompts, answers, tasks, and infos
        """
        batches = self._async_dataloader.peek_ahead(batch_offset)
        batch = batches[batch_offset - 1] if batch_offset > 0 else batches[0]
        if isinstance(batch, dict):
            batch = [batch]

        # Gather batch data from all processes (support legacy & new field names)
        def _get_field(item: Dict[str, Any], *names, default=None):
            for n in names:
                if n in item:
                    return item[n]
            return default

        prompts = [_get_field(x, 'prompt', 'text', default=x) for x in batch]

        # "answer" is optional in many datasets (e.g., unsupervised tasks). Fallback to using the input text.
        answers = [_get_field(x, 'answer', 'text', default=None) for x in batch]

        tasks = [x.get('task', 'default') for x in batch]
        infos = [x.get('info', {}) for x in batch]
        all_prompts = gather_object(prompts)
        all_answers = gather_object(answers)
        all_tasks = gather_object(tasks)
        all_infos = gather_object(infos)
        return BatchData(
            prompts=all_prompts,
            answers=all_answers, 
            tasks=all_tasks,
            infos=all_infos
        )

    def _prepare_inputs(  # type: ignore
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Modular training loop using configurable pipeline strategies.
        
        This method orchestrates the training process through pluggable components:
        1. Training orchestration (when to generate, sync weights, etc.)
        2. Weight synchronization strategy  
        3. Batch data gathering strategy
        4. Batch processing strategy
        5. Input buffering strategy
        """
        self.logger.push_debug("Prepare")
        
        # Ensure all processes are synchronized at the start
        self.accelerator.wait_for_everyone()

        # Check if we need to generate new completions using orchestrator
        if self.training_pipeline.orchestrator.should_generate_new_batch(self._step, self._buffered_inputs):
            self.logger.push_info("Generate")
            
            # Sync weights using strategy
            if self.training_pipeline.weight_sync_strategy.should_sync_weights(
                self.state.global_step, self._last_loaded_step
            ):
                self.logger.info(f"Syncing weights to vLLM at step {self.state.global_step}")
                self.training_pipeline.weight_sync_strategy.sync_weights(self)
                self._last_loaded_step = self.state.global_step

            # Start async generator if not started
            if not self._async_started and self.accelerator.is_main_process:
                self.async_generator.start()
            self._async_started = True
            self.accelerator.wait_for_everyone()

            # Plan generation phase using orchestrator
            generation_phase = self.training_pipeline.orchestrator.plan_generation_phase(
                self._step, self.async_generator
            )

            # Submit batches planned by orchestrator
            batches_submitted = 0

            self.logger.push_debug("Submit")
            for batch_id in range(self._next_batch_id, generation_phase.target_batch_id + 1):
                batch_offset = batch_id - generation_phase.batch_id_to_retrieve
                batch_data = self._gather_batch_data(batch_offset)

                local_batch_size = len(batch_data.prompts) // self.accelerator.num_processes

                # Submit batch (main process only)
                if self.accelerator.is_main_process:
                    batch_id = self.state.global_step
                    
                    with LogContext(f"📦 Preparing Batch {batch_id}...", f"batch_{batch_id}", logger=self.logger):
                        self.logger.info(f"[cyan]Batch Configuration:[/]")
                        self.logger.info(f"  • Global step: [green]{self.state.global_step}[/]")
                        self.logger.info(f"  • Total prompts: [green]{len(batch_data.prompts)}[/]")
                        self.logger.info(f"  • Processes: [green]{self.accelerator.num_processes}[/]")
                        self.logger.info(f"  • Local batch size: [green]{local_batch_size}[/]")

                        # Create configuration objects
                        from datasets import Dataset
                        rows_dataset = Dataset.from_dict({
                            'prompt': batch_data.prompts, 
                            'text': batch_data.prompts, 
                            'answer': batch_data.answers, 
                            'task': batch_data.tasks, 
                            'info': batch_data.infos
                        })
                        
                        generation_config = GenerationConfig(
                            max_completion_length=self.max_completion_length,
                            mask_truncated_completions=self.mask_truncated_completions,
                            mask_env_responses=self.mask_env_responses,
                            max_concurrent=self.max_concurrent,
                            generation_timeout=self.async_generator.generation_timeout,
                            num_batches_ahead=self.async_generator.num_batches_ahead,
                        )
                        
                        processing_config = ProcessingConfig(
                            processing_class=self.processing_class,  # type: ignore
                            mask_env_responses=self.mask_env_responses,
                        )
                        
                        distributed_config = DistributedConfig(
                            device=self.accelerator.device,
                            accelerator=self.accelerator,
                            process_index=self.accelerator.process_index,
                            num_processes=self.accelerator.num_processes,
                            local_batch_size=local_batch_size,
                        )
                        
                        request = BatchRequest(
                            batch_id=batch_id,
                            rows=rows_dataset,
                            generation_config=generation_config,
                            processing_config=processing_config,
                            distributed_config=distributed_config,
                        )
                        self.async_generator.submit_batch(request)
                        self.logger.info(f"[green]✓[/] Batch {batch_id} submitted to async generator")
                self.accelerator.wait_for_everyone()
            self.logger.pop()

            # Update next batch id
            if self.accelerator.is_main_process:
                self._next_batch_id = self._next_batch_id + batches_submitted
                if batches_submitted > 0:
                    self.logger.info(f"Submitted {batches_submitted} batches, next_batch_id now {self._next_batch_id}")
            self.accelerator.wait_for_everyone()
            # Synchronize next_batch_id across all processes
            next_batch_id_list = [self._next_batch_id if self.accelerator.is_main_process else 0]
            broadcast_object_list(next_batch_id_list, from_process=0)
            self._next_batch_id = next_batch_id_list[0]
            self.accelerator.wait_for_everyone()

            # Now retrieve the batch we need for this step
            if self.accelerator.is_main_process:
                self.logger.info(f"Retrieving batch {generation_phase.batch_id_to_retrieve} for processing")

                # Get batch result
                batch_result = self.async_generator.get_batch(generation_phase.batch_id_to_retrieve)
                processed_results = batch_result.processed_results

                # Package raw data for broadcast (not tensors yet)
                broadcast_data = {
                    'prompt_ids':      processed_results['prompt_ids'],
                    'prompt_mask':     processed_results['prompt_mask'],
                    'completion_ids':  processed_results['completion_ids'],
                    'completion_mask': processed_results['completion_mask'],
                    'rewards':         processed_results['rewards'],
                    'all_reward_dict': batch_result.all_reward_dict if hasattr(batch_result, 'all_reward_dict') else {'reward': processed_results['rewards']},
                    'completions':     batch_result.completions if hasattr(batch_result, 'completions') else [],
                    'prompts':         batch_result.prompts if hasattr(batch_result, 'prompts') else [],
                }
            else:
                broadcast_data = None
            self.accelerator.wait_for_everyone()

            self.logger.push_debug("Broadcast")
            # Broadcast processed data
            broadcast_list = [broadcast_data]
            broadcast_object_list(broadcast_list, from_process=0)
            broadcast_data = broadcast_list[0]
            self.accelerator.wait_for_everyone()
            self.logger.pop()

            # Each process takes its slice
            process_slice = slice(
                self.accelerator.process_index * len(inputs),
                (self.accelerator.process_index + 1) * len(inputs),
            )

            self.logger.push_debug("Process")
            # Create rewards tensor and compute advantages using full batch
            assert broadcast_data is not None  # After broadcast, all processes have data
            all_rewards = torch.tensor(broadcast_data['rewards'], device=self.accelerator.device)
            all_advantages = self._compute_advantages(all_rewards)

            # Now create tensors only for this process's slice
            prompt_ids_list = []
            prompt_mask_list = []
            completion_ids_list = []
            completion_mask_list = []

            for i in range(process_slice.start, process_slice.stop):
                prompt_ids_list.append(torch.tensor(broadcast_data['prompt_ids'][i], device=self.accelerator.device))
                prompt_mask_list.append(torch.tensor(broadcast_data['prompt_mask'][i], device=self.accelerator.device))
                completion_ids_list.append(torch.tensor(broadcast_data['completion_ids'][i], device=self.accelerator.device))
                completion_mask_list.append(torch.tensor(broadcast_data['completion_mask'][i], device=self.accelerator.device))

            # Pad sequences
            prompt_ids = pad(prompt_ids_list, padding_value=self.processing_class.pad_token_id, padding_side='left')  # type: ignore
            prompt_mask = pad(prompt_mask_list, padding_side='left')  # type: ignore
            completion_ids = pad(completion_ids_list, padding_value=self.processing_class.pad_token_id, padding_side='right')  # type: ignore
            completion_mask = pad(completion_mask_list)

            # Truncate if needed
            if self.max_prompt_length is not None and prompt_ids.size(1) > self.max_prompt_length:
                prompt_ids = prompt_ids[:, -self.max_prompt_length:]
                prompt_mask = prompt_mask[:, -self.max_prompt_length:]

            if self.max_completion_length is not None and completion_ids.size(1) > self.max_completion_length:
                completion_ids = completion_ids[:, :self.max_completion_length]
                completion_mask = completion_mask[:, :self.max_completion_length]

            # Take this process's slice of advantages
            advantages = all_advantages[process_slice]

            # Log metrics on main process only
            if self.accelerator.is_main_process:
                self._log_reward_metrics_primary(
                    mode="train",
                    all_reward_dict=broadcast_data['all_reward_dict'],
                    all_rewards=all_rewards,
                    generation_batch_size=len(all_rewards)
                )

                self._log_textual_data_primary(
                    all_prompts=broadcast_data['prompts'],
                    all_completions=broadcast_data['completions'],
                    all_reward_dict=broadcast_data['all_reward_dict']
                )

                # Log completion metrics using full batch data on CPU to save memory
                self._log_completion_metrics_primary(
                    mode="train",
                    all_completion_mask=broadcast_data['completion_mask'],
                    all_completion_ids=broadcast_data['completion_ids'],
                    all_prompt_mask=broadcast_data['prompt_mask']
                )

            # Concatenate all data for shuffling
            full_batch = {
                "prompt_ids":          prompt_ids,
                "prompt_mask":         prompt_mask,
                "completion_ids":      completion_ids,
                "completion_mask":     completion_mask,
                "old_per_token_logps": None,
                "advantages":          advantages,
            }

            # Shuffle and split for gradient accumulation
            full_batch = shuffle_tensor_dict(full_batch)
            self._buffered_inputs = split_tensor_dict(full_batch, self.gradient_accumulation_steps)
            self.accelerator.wait_for_everyone()
            self.logger.pop()
            self.logger.pop()
            
        # Return appropriate slice from buffer using strategy
        result = self.training_pipeline.input_buffer_strategy.get_step_inputs(self, self._step)
        self._step += 1
        self.accelerator.wait_for_everyone()
        
        self.logger.pop()
        return result
    
    def _process_batch_result_impl(self, batch_result: Any, process_slice: slice) -> Dict[str, torch.Tensor]:
        """Implementation method for processing batch results"""
        # Extract processed results  
        processed_results = batch_result.processed_results

        # Package raw data for broadcast (not tensors yet)
        broadcast_data = {
            'prompt_ids':      processed_results['prompt_ids'],
            'prompt_mask':     processed_results['prompt_mask'],
            'completion_ids':  processed_results['completion_ids'],
            'completion_mask': processed_results['completion_mask'],
            'rewards':         processed_results.get('rewards', []),
            'all_reward_dict': batch_result.all_reward_dict if hasattr(batch_result, 'all_reward_dict') else {'reward': processed_results.get('rewards', [])},
            'completions':     batch_result.completions if hasattr(batch_result, 'completions') else [],
            'prompts':         batch_result.prompts if hasattr(batch_result, 'prompts') else [],
        }
        
        # Create rewards tensor and compute advantages using full batch
        all_rewards = torch.tensor(broadcast_data['rewards'], device=self.accelerator.device)
        all_advantages = self._compute_advantages(all_rewards)

        # Now create tensors only for this process's slice
        prompt_ids_list = []
        prompt_mask_list = []
        completion_ids_list = []
        completion_mask_list = []

        for i in range(process_slice.start, process_slice.stop):
            prompt_ids_list.append(torch.tensor(broadcast_data['prompt_ids'][i], device=self.accelerator.device))
            prompt_mask_list.append(torch.tensor(broadcast_data['prompt_mask'][i], device=self.accelerator.device))
            completion_ids_list.append(torch.tensor(broadcast_data['completion_ids'][i], device=self.accelerator.device))
            completion_mask_list.append(torch.tensor(broadcast_data['completion_mask'][i], device=self.accelerator.device))

        # Pad sequences
        prompt_ids = pad(prompt_ids_list, padding_value=self.processing_class.pad_token_id, padding_side='left')  # type: ignore
        prompt_mask = pad(prompt_mask_list, padding_side='left')  # type: ignore
        completion_ids = pad(completion_ids_list, padding_value=self.processing_class.pad_token_id, padding_side='right')  # type: ignore
        completion_mask = pad(completion_mask_list)

        # Truncate if needed
        if self.max_prompt_length is not None and prompt_ids.size(1) > self.max_prompt_length:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        if self.max_completion_length is not None and completion_ids.size(1) > self.max_completion_length:
            completion_ids = completion_ids[:, :self.max_completion_length]
            completion_mask = completion_mask[:, :self.max_completion_length]

        # Take this process's slice of advantages
        advantages = all_advantages[process_slice]
        
        return {
            "prompt_ids":          prompt_ids,
            "prompt_mask":         prompt_mask,
            "completion_ids":      completion_ids,
            "completion_mask":     completion_mask,
            "old_per_token_logps": None,
            "advantages":          advantages,
        }

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute advantages from rewards using algorithm-specific logic."""
        return self.algorithm.compute_advantages(rewards, self.num_generations)


    def compute_loss(self,
                     model: PreTrainedModel,
                     inputs: Dict[str, torch.Tensor],
                     return_outputs: bool = False,
                     num_items_in_batch: int | None = None) -> torch.Tensor:
        self.logger.push_debug("Loss")
        
        mode = "train"
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        
        self.logger.push_debug("Logprobs")
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        self.logger.pop()

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps,
        # so we can skip it's computation (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        
        self.logger.push_debug("Policy")
        # Compute importance ratios using algorithm-specific logic
        importance_ratios = self.algorithm.compute_importance_ratios(
            per_token_logps, old_per_token_logps, completion_mask
        )
        
        # Compute policy loss using algorithm-specific logic
        rl_result = self.algorithm.compute_policy_loss(
            importance_ratios, advantages, completion_mask
        )
        policy_loss = rl_result.loss
        self.logger.pop()

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            self.logger.push_debug("KL")
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():  # type: ignore
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, input_ids, attention_mask, logits_to_keep
                        )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            # Add KL penalty to policy loss
            per_token_kl_loss = self.beta * per_token_kl
            if hasattr(rl_result, 'loss'):
                # For algorithms that return token-level losses, add KL to each token
                if len(per_token_kl_loss.shape) == 2:  # token-level
                    total_loss = policy_loss + (per_token_kl_loss * completion_mask).sum() / completion_mask.sum()
                else:  # sequence-level
                    total_loss = policy_loss + per_token_kl_loss.mean()
            else:
                total_loss = policy_loss + per_token_kl_loss.mean()
            
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            gathered_kl = self.accelerator.gather_for_metrics(mean_kl)
            kl_value = gathered_kl.nanmean().item() if hasattr(gathered_kl, 'nanmean') else float(gathered_kl)
            self._metrics.get_mode_metrics(mode).algorithm.add_kl_metric(kl_value)
            self.logger.pop()
        else:
            total_loss = policy_loss

        # Log algorithm-specific metrics
        algorithm_name = self.algorithm.get_algorithm_name()
        mode_metrics = self._metrics.get_mode_metrics(mode)
        
        for metric_name, metric_value in rl_result.metrics.items():
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()
            mode_metrics.algorithm.add_algorithm_metric(algorithm_name, metric_name, metric_value)
        
        # Log clipping statistics
        mode_metrics.clipping.add_clipping_stats(rl_result.clipping_stats, self.accelerator)
        
        self.logger.pop()
        return total_loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        """
        Override the evaluate method to use env.evaluate() directly.
        This bypasses the standard batch-by-batch evaluation and uses the environment's
        built-in evaluation logic instead.
        """
        self.logger.push_info("Eval")
        self.logger.info("Running evaluation using environment's evaluate method")

        # Call env.evaluate with appropriate parameters
        eval_results = self.loom.test(
            client=self.oai_client,
            model=self._get_model_name(),
            sampling_args=self._get_sampling_args(),
            max_concurrent=self.max_concurrent,
        )

        # Process results to compute metrics
        metrics = {}

        # Compute reward statistics
        if 'reward' in eval_results:
            rewards = torch.tensor(eval_results['reward'])
            metrics['eval_reward'] = rewards.mean().item()
            metrics['eval_reward_std'] = rewards.std().item()

        # Log individual attraction rule gravities
        for key in eval_results:
            if key.startswith('reward_') and key != 'reward':
                reward_values = eval_results[key]
                if isinstance(reward_values, list):
                    metrics[f'eval_rewards/{key[7:]}'] = float(np.mean(reward_values))
                else:
                    metrics[f'eval_rewards/{key[7:]}'] = reward_values.mean().item()

        # Compute completion length statistics
        if 'completion' in eval_results:
            completions = eval_results['completion']
            if isinstance(completions[0], str):
                # Completion format - directly tokenize strings
                completion_lengths = [len(self.processing_class.encode(c)) for c in completions]  # type: ignore
            else:
                # Chat format - use apply_chat_template
                completion_lengths = []
                for comp in completions:
                    # Apply chat template to get the full text
                    tokens = self.processing_class.apply_chat_template(comp, tokenize=True, add_generation_prompt=False)  # type: ignore
                    # Tokenize and count
                    completion_lengths.append(len(tokens))

            metrics['eval_completions/mean_length'] = float(np.mean(completion_lengths))
            metrics['eval_completions/min_length'] = int(np.min(completion_lengths))
            metrics['eval_completions/max_length'] = int(np.max(completion_lengths))

        # Log sample completions if requested
        if self.accelerator.is_main_process and self.log_completions and 'prompt' in eval_results:
            # Prepare textual logs
            prompts = eval_results['prompt'][:self.num_completions_to_print]
            completions = eval_results['completion'][:self.num_completions_to_print]

            # Extract rewards for logging
            reward_dict = {}
            if 'reward' in eval_results:
                reward_dict['reward'] = eval_results['reward'][:self.num_completions_to_print]
            for key in eval_results:
                if key.startswith('reward_') and key != 'reward':
                    reward_dict[key] = eval_results[key][:self.num_completions_to_print]

            # Print sample
            print_prompt_completions_sample(
                prompts,
                completions,
                reward_dict,
                self.state.global_step,
            )

            # Log to wandb if available
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table_data = {
                    "step":       [str(self.state.global_step)] * len(prompts),
                    "prompt":     prompts,
                    "completion": completions,
                }
                for k, v in reward_dict.items():
                    table_data[k] = v

                df = pd.DataFrame(table_data)
                wandb.log({"eval_completions": wandb.Table(dataframe=df)})

        # Log all metrics
        self.log(metrics)

        # Return metrics dict to match base class signature
        self.logger.pop()
        return metrics

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model is not None and self.model.training else "eval"  # type: ignore
        
        # Get structured metrics and convert to dictionary
        mode_metrics_dict = self._metrics.to_dict(mode)
        metrics = {}
        
        # Average the metrics (each metric is a list of values)
        for key, val in mode_metrics_dict.items():
            if isinstance(val, list) and len(val) > 0:
                metrics[key] = sum(val) / len(val)

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear_mode(mode)

        if self.accelerator.is_main_process and self.log_completions:
            if not self._textual_logs.is_empty():
                print_prompt_completions_sample(
                    list(self._textual_logs.prompts),
                    list(self._textual_logs.completions),
                    {k: list(v) for k, v in self._textual_logs.rewards.items()},
                    self.state.global_step,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step":       [str(self.state.global_step)] * len(self._textual_logs.prompts),
                    "prompt":     list(self._textual_logs.prompts),
                    "completion": list(self._textual_logs.completions),
                    **{k: list(v) for k, v in self._textual_logs.rewards.items()},
                }
                if len(table["prompt"]) > 0:
                    df = pd.DataFrame(table)
                    if self.wandb_log_unique_prompts:
                        df = df.drop_duplicates(subset=["prompt"])
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            # Clear the textual logs after logging
            self._textual_logs.clear()

    def _log_reward_metrics_primary(
        self,
        mode: str,
        all_reward_dict: Dict[str, Any],
        all_rewards: torch.Tensor,
        generation_batch_size: int
    ) -> None:
        """
        Log generation metrics (PRIMARY PROCESS ONLY).
        This handles reward statistics and per-reward-function metrics using the full batch data.
        """
        mode_metrics = self._metrics.get_mode_metrics(mode)
        
        # Log reward statistics using full batch
        mean_rewards = all_rewards.view(-1, self.num_generations).mean(dim=1)
        std_rewards = all_rewards.view(-1, self.num_generations).std(dim=1)
        mode_metrics.rewards.add_reward_stats(mean_rewards.mean().item(), std_rewards.mean().item())

        # Log individual attraction rule gravities as metrics
        for reward_key in all_reward_dict:
            if reward_key != 'reward':  # Skip the consolidated reward
                reward_values = all_reward_dict[reward_key]
                if isinstance(reward_values, list):
                    reward_tensor = torch.tensor(reward_values, device=all_rewards.device)
                else:
                    reward_tensor = reward_values
                mean_reward = reward_tensor.mean().item()
                mode_metrics.rewards.add_individual_reward(reward_key, mean_reward)

    def _log_textual_data_primary(
        self,
        all_prompts: List[Union[str, List[Dict[str, Any]]]],
        all_completions: List[Union[str, List[Dict[str, Any]]]],
        all_reward_dict: Dict[str, Any]
    ) -> None:
        """
        Log textual data for wandb (PRIMARY PROCESS ONLY).
        This logs the full batch of prompts, completions, and rewards.
        """
        self._textual_logs.add_logs(all_prompts, all_completions, all_reward_dict)

    def _log_completion_metrics_primary(
        self,
        mode: str,
        all_completion_mask: List[List[int]],
        all_completion_ids: List[List[int]],
        all_prompt_mask: List[List[int]]
    ) -> None:
        """
        Log completion-related metrics (PRIMARY PROCESS ONLY).
        This handles completion length statistics using the full batch data.
        """
        mode_metrics = self._metrics.get_mode_metrics(mode)
        
        # Log token count
        if mode == "train":
            total_tokens = sum(len(pm) + len(cm) for pm, cm in zip(all_prompt_mask, all_completion_mask))
            self.state.num_input_tokens_seen += total_tokens
        mode_metrics.num_tokens = [self.state.num_input_tokens_seen]

        # Log completion lengths
        completion_lengths = [sum(mask) for mask in all_completion_mask]
        mode_metrics.completions.add_length_stats(completion_lengths)

        # Check for EOS tokens
        term_lengths = []
        for comp_ids, comp_mask in zip(all_completion_ids, all_completion_mask):
            has_eos = any(token == self.processing_class.eos_token_id for token, mask in zip(comp_ids, comp_mask) if mask)  # type: ignore
            if has_eos:
                term_lengths.append(sum(comp_mask))

        mode_metrics.completions.add_termination_stats(term_lengths, len(completion_lengths))

def print_prompt_completions_sample(
    prompts: list[str],
    completions: list[dict],
    rewards: dict[str, list[float]],
    step: int,
    num_samples: int = 1,  # Number of samples to display
) -> None:
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    table.add_column("Reward", style="bold cyan", justify="right")

    # Get the reward values from the dictionary
    reward_values = rewards.get("reward", [])

    # Ensure we have rewards for all prompts/completions
    if len(reward_values) < len(prompts):
        # Pad with zeros if we don't have enough rewards
        reward_values = reward_values + [0.0] * (len(prompts) - len(reward_values))

    # Only show the first num_samples samples
    samples_to_show = min(num_samples, len(prompts))

    for i in range(samples_to_show):
        prompt = prompts[i]
        completion = completions[i]
        reward = reward_values[i]

        # Format prompt (can be string or list of dicts)
        formatted_prompt = Text()
        if isinstance(prompt, str):
            formatted_prompt = Text(prompt)
        elif isinstance(prompt, list):
            # For chat format, only show the last message content (typically the user's question)
            if prompt:
                last_message = prompt[-1]
                content = last_message.get("content", "")
                formatted_prompt = Text(content, style="bright_yellow")
            else:
                formatted_prompt = Text("")
        else:
            formatted_prompt = Text(str(prompt))

        # Create a formatted Text object for completion with alternating colors based on role
        formatted_completion = Text()

        if isinstance(completion, dict):
            # Handle single message dict
            role = completion.get("role", "")
            content = completion.get("content", "")
            style = "bright_cyan" if role == "assistant" else "bright_magenta"
            formatted_completion.append(f"{role}: ", style="bold")
            formatted_completion.append(content, style=style)
        elif isinstance(completion, list):
            # Handle list of message dicts
            for i, message in enumerate(completion):
                if i > 0:
                    formatted_completion.append("\n\n")

                role = message.get("role", "")
                content = message.get("content", "")

                # Set style based on role
                style = "bright_cyan" if role == "assistant" else "bright_magenta"

                formatted_completion.append(f"{role}: ", style="bold")
                formatted_completion.append(content, style=style)
        else:
            # Fallback for string completions
            formatted_completion = Text(str(completion))

        table.add_row(formatted_prompt, formatted_completion, Text(f"{reward:.2f}"))
        if i < samples_to_show - 1:  # Don't add section after last row
            pass  # table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)

# Factory functions for creating RL trainers with different algorithms

def create_grpo_trainer(
    loom: Loom,
    model: PreTrainedModel,
    args: GRPOConfig,
    tokenizer: PreTrainedTokenizerBase,
    callbacks: Optional[list[TrainerCallback]] = None,
    optimizers: Optional[OptimizerConfig] = None,
    peft_config: Optional[PeftConfig] = None
) -> GRPOTrainer:
    """Factory function to create a GRPO trainer"""
    algorithm = GRPOAlgorithm(config=args, logger=log.getLogger('errloom.training.GRPOAlgorithm'))
    return GRPOTrainer(
        loom=loom,
        model=model,
        args=args,
        tokenizer=tokenizer,
        callbacks=callbacks,
        optimizers=optimizers,
        peft_config=peft_config,
        algorithm=algorithm
    )

def create_gspo_trainer(
    loom: Loom,
    model: PreTrainedModel,
    args: GRPOConfig,  # Can reuse GRPOConfig, GSPO uses different epsilon values
    tokenizer: PreTrainedTokenizerBase,
    callbacks: Optional[list[TrainerCallback]] = None,
    optimizers: Optional[OptimizerConfig] = None,
    peft_config: Optional[PeftConfig] = None
) -> GRPOTrainer:
    """Factory function to create a GSPO trainer"""
    algorithm = GSPOAlgorithm(config=args, logger=log.getLogger('errloom.training.GSPOAlgorithm'))
    return GRPOTrainer(
        loom=loom,
        model=model,
        args=args,
        tokenizer=tokenizer,
        callbacks=callbacks,
        optimizers=optimizers,
        peft_config=peft_config,
        algorithm=algorithm
    )

class GSPOTrainer(GRPOTrainer):
    """GSPO (Group Sequence Policy Optimization) trainer - convenience class"""
    
    def __init__(
        self,
        loom: Loom,
        model: PreTrainedModel,
        args: GRPOConfig,
        tokenizer: PreTrainedTokenizerBase,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: Optional[OptimizerConfig] = None,
        peft_config: Optional[PeftConfig] = None
    ):
        """Initialize GSPO trainer with GSPO algorithm"""
        algorithm = GSPOAlgorithm(config=args, logger=log.getLogger('errloom.training.GSPOAlgorithm'))
        super().__init__(
            loom=loom,
            model=model,
            args=args,
            tokenizer=tokenizer,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            algorithm=algorithm
        )

@dataclass
class TrainingStep:
    """Container for a single training step data"""
    step_id: int
    batch_data: BatchData
    processed_inputs: Dict[str, torch.Tensor]
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass  
class GenerationPhase:
    """Container for generation phase data"""
    should_generate: bool
    batch_id_to_retrieve: int
    target_batch_id: int
    batches_to_submit: List[int]

class TrainingOrchestrator(ABC):
    """Abstract orchestrator for training phases
    
    🎪 Advanced Use Cases:
    - CurriculumOrchestrator: Difficulty progression based on performance metrics
    - AdaptiveOrchestrator: Dynamic batch scheduling based on loss convergence
    - MultiObjectiveOrchestrator: Coordinated sampling for Pareto frontier exploration
    - OnlineOrchestrator: Continuous adaptation for streaming data environments
    - FederatedOrchestrator: Cross-client coordination for federated RL
    
    🔮 Future Algorithms:
    - GHPOOrchestrator: Difficulty-aware scheduling for Group Hindsight Policy Optimization
    - SPOOrchestrator: Self-play coordination for tournament-style preference optimization
    - MetaOrchestrator: Meta-learning orchestration for few-shot adaptation
    - HierarchicalOrchestrator: Multi-level coordination for hierarchical RL
    """
    
    @abstractmethod
    def should_generate_new_batch(self, step: int, buffered_inputs: Any) -> bool:
        """Determine if we need to generate new completions
        
        Use cases:
        - Curriculum: Generate based on performance thresholds
        - Adaptive: Generate based on convergence metrics  
        - Online: Generate based on data stream availability
        - Memory-aware: Generate based on memory pressure
        """
        pass
    
    @abstractmethod
    def plan_generation_phase(self, step: int, async_generator: Any) -> GenerationPhase:
        """Plan which batches to submit and retrieve
        
        Use cases:
        - Multi-turn: Plan conversation-aware batch sequences
        - Distributed: Coordinate across geographical regions
        - Federated: Plan client-specific batch distributions
        - Hierarchical: Plan multi-level batch coordination
        """
        pass

class WeightSyncStrategy(ABC):
    """Abstract strategy for syncing model weights to inference engine
    
    🎪 Advanced Use Cases:
    - MultiClusterWeightSync: Sync to multiple geographical clusters
    - FederatedWeightSync: Aggregate and distribute federated updates
    - HierarchicalWeightSync: Multi-tier sync for edge/cloud deployments
    - StreamingWeightSync: Real-time continuous updates for online learning
    - MemoryEfficientWeightSync: Gradient checkpointing for large models
    
    🔮 Future Deployments:
    - QuantizedWeightSync: Dynamic precision adjustment for edge devices
    - SparsityAwareSync: Efficient sync for sparse/MoE models
    - VersionedWeightSync: A/B testing with multiple model versions
    - BlockchainWeightSync: Decentralized model distribution
    """
    
    @abstractmethod
    def should_sync_weights(self, current_step: int, last_synced_step: int) -> bool:
        """Determine if weights should be synced
        
        Use cases:
        - Adaptive: Sync based on weight divergence metrics
        - Federated: Sync based on client aggregation schedules
        - Memory-aware: Sync based on available bandwidth/memory
        - Performance-driven: Sync based on inference latency targets
        """
        pass
    
    @abstractmethod
    def sync_weights(self, trainer: 'GRPOTrainer') -> None:
        """Perform weight synchronization
        
        Use cases:
        - Multi-cluster: Parallel sync to multiple inference engines
        - Federated: Secure aggregation and distribution
        - Streaming: Incremental delta updates
        - Quantized: Dynamic precision adjustment during sync
        """
        pass

class BatchDataStrategy(ABC):
    """Abstract strategy for gathering batch data
    
    🎪 Advanced Use Cases:
    - StreamingBatchData: Real-time data from live user interactions
    - CurriculumBatchData: Difficulty-progressive data selection
    - MultiModalBatchData: Coordinated text/image/audio data gathering
    - DistributedBatchData: Cross-datacenter data federation
    - MemoryMappedBatchData: Zero-copy data loading for massive datasets
    
    🔮 Future Data Sources:
    - BlockchainBatchData: Decentralized training data marketplace
    - SyntheticBatchData: AI-generated training data with quality control
    - PrivacyPreservingBatchData: Differential privacy for sensitive data
    - MultiAgentBatchData: Coordinated data from multiple AI agents
    """
    
    @abstractmethod
    def gather_batch_data(self, trainer: 'GRPOTrainer', batch_offset: int) -> BatchData:
        """Gather batch data from distributed processes
        
        Use cases:
        - Streaming: Gather from real-time data streams
        - Curriculum: Select data based on difficulty metrics
        - Multi-modal: Coordinate heterogeneous data types
        - Privacy-aware: Apply differential privacy during gathering
        """
        pass

class BatchProcessingStrategy(ABC):
    """Abstract strategy for processing generated batches
    
    🎪 Advanced Use Cases:
    - MultiTurnBatchProcessing: Conversation-level reward computation
    - HierarchicalBatchProcessing: Multi-level action space processing
    - MultiObjectiveBatchProcessing: Pareto-efficient reward aggregation
    - PrivacyPreservingBatchProcessing: Federated learning with secure aggregation
    - MixedPrecisionBatchProcessing: Dynamic precision for memory efficiency
    
    🔮 Future Processing Paradigms:
    - NeuralProcessing: Learned batch processing with neural networks
    - QuantumProcessing: Quantum advantage for optimization problems
    - CausalProcessing: Causal inference for robust reward attribution
    - MetaProcessing: Few-shot adaptation of processing strategies
    """
    
    @abstractmethod  
    def process_batch_result(self, 
                           trainer: 'GRPOTrainer',
                           batch_result: Any,
                           process_slice: slice) -> Dict[str, torch.Tensor]:
        """Process batch results into training inputs
        
        Use cases:
        - Multi-turn: Process conversation contexts with turn-level rewards
        - Hierarchical: Process multi-level action sequences
        - Multi-objective: Compute multiple reward functions simultaneously
        - Causal: Apply causal masking for robust attribution
        """
        pass

class InputBufferStrategy(ABC):
    """Abstract strategy for managing input buffering across training steps
    
    🎪 Advanced Use Cases:
    - GradientCheckpointingBuffer: Memory-efficient training for large models
    - AdaptiveBatchSizeBuffer: Dynamic batch sizing based on memory/performance
    - PriorityExperienceBuffer: Importance sampling for experience replay
    - StreamingBuffer: Continuous data flow for online learning
    - FederatedBuffer: Coordinated buffering across federated clients
    
    🔮 Future Buffer Architectures:
    - QuantumBuffer: Quantum superposition for parallel hypothesis testing
    - NeuralBuffer: Learned compression for efficient memory usage
    - CausalBuffer: Temporally-aware buffering for causal inference
    - MetaBuffer: Adaptive buffering strategies learned via meta-learning
    """
    
    @abstractmethod
    def buffer_inputs(self, 
                     trainer: 'GRPOTrainer',
                     processed_inputs: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Buffer and split inputs for gradient accumulation
        
        Use cases:
        - Memory-efficient: Apply gradient checkpointing during buffering
        - Adaptive: Dynamically adjust buffer size based on memory pressure
        - Priority-based: Buffer high-importance samples preferentially
        - Streaming: Maintain rolling buffer for continuous learning
        """
        pass
    
    @abstractmethod
    def get_step_inputs(self, 
                       trainer: 'GRPOTrainer',
                       step: int) -> Dict[str, torch.Tensor]:
        """Get inputs for current step from buffer
        
        Use cases:
        - Priority sampling: Return high-importance samples first
        - Curriculum: Return difficulty-appropriate samples
        - Multi-objective: Return samples balancing multiple objectives
        - Federated: Return client-specific sample distributions
        """
        pass

@dataclass
class TrainingPipeline:
    """Configurable training pipeline with pluggable strategies
    
    🎪 Complete Advanced Pipelines:
    
    Multi-Turn Conversational RL:
    ```python
    pipeline = TrainingPipeline(
        orchestrator=ConversationOrchestrator(turn_limit=10),
        weight_sync_strategy=VLLMWeightSyncStrategy(),
        batch_data_strategy=ConversationBatchDataStrategy(),
        batch_processing_strategy=TurnAwareBatchProcessing(),
        input_buffer_strategy=ConversationBuffer()
    )
    ```
    
    Federated Learning:
    ```python
    pipeline = TrainingPipeline(
        orchestrator=FederatedOrchestrator(aggregation_rounds=100),
        weight_sync_strategy=FederatedWeightSync(clients=client_list),
        batch_data_strategy=PrivacyPreservingBatchData(epsilon=1.0),
        batch_processing_strategy=SecureAggregationProcessing(),
        input_buffer_strategy=FederatedBuffer(dp_noise=0.1)
    )
    ```
    
    Curriculum Learning:
    ```python
    pipeline = TrainingPipeline(
        orchestrator=CurriculumOrchestrator(difficulty_schedule="linear"),
        weight_sync_strategy=AdaptiveWeightSync(sync_threshold=0.01),
        batch_data_strategy=DifficultyAwareBatchData(),
        batch_processing_strategy=ProgressiveBatchProcessing(),
        input_buffer_strategy=CurriculumBuffer(difficulty_sampling=True)
    )
    ```
    
    Online/Streaming Learning:
    ```python
    pipeline = TrainingPipeline(
        orchestrator=StreamingOrchestrator(window_size=1000),
        weight_sync_strategy=RealTimeWeightSync(latency_target_ms=50),
        batch_data_strategy=StreamingBatchData(kafka_topics=["user_feedback"]),
        batch_processing_strategy=IncrementalBatchProcessing(),
        input_buffer_strategy=StreamingBuffer(decay_factor=0.99)
    )
    ```
    
    🔮 Future Algorithm Pipelines:
    
    GHPO (Group Hindsight Policy Optimization):
    ```python
    pipeline = TrainingPipeline(
        orchestrator=GHPOOrchestrator(hindsight_horizon=50),
        weight_sync_strategy=VLLMWeightSyncStrategy(),
        batch_data_strategy=HindsightBatchData(),
        batch_processing_strategy=CounterfactualProcessing(),
        input_buffer_strategy=HindsightBuffer()
    )
    ```
    
    SPO (Self-Play Preference Optimization):
    ```python
    pipeline = TrainingPipeline(
        orchestrator=SelfPlayOrchestrator(tournament_size=64),
        weight_sync_strategy=MultiVersionWeightSync(),
        batch_data_strategy=TournamentBatchData(),
        batch_processing_strategy=PreferenceProcessing(),
        input_buffer_strategy=TournamentBuffer()
    )
    ```
    
    Multi-Objective RL:
    ```python
    pipeline = TrainingPipeline(
        orchestrator=ParetoOrchestrator(objectives=["helpfulness", "safety", "efficiency"]),
        weight_sync_strategy=VLLMWeightSyncStrategy(),
        batch_data_strategy=MultiObjectiveBatchData(),
        batch_processing_strategy=ParetoProcessing(),
        input_buffer_strategy=ParetoBuffer()
    )
    ```
    
    Quantum-Enhanced RL:
    ```python
    pipeline = TrainingPipeline(
        orchestrator=QuantumOrchestrator(qubit_count=1024),
        weight_sync_strategy=QuantumWeightSync(),
        batch_data_strategy=QuantumBatchData(),
        batch_processing_strategy=QuantumProcessing(),
        input_buffer_strategy=QuantumBuffer()
    )
    ```
    """
    orchestrator: TrainingOrchestrator
    weight_sync_strategy: WeightSyncStrategy  
    batch_data_strategy: BatchDataStrategy
    batch_processing_strategy: BatchProcessingStrategy
    input_buffer_strategy: InputBufferStrategy

@dataclass
class RewardMetrics:
    """Container for reward-related metrics"""
    reward: List[float] = field(default_factory=list)
    reward_std: List[float] = field(default_factory=list)
    individual_rewards: Dict[str, List[float]] = field(default_factory=dict)
    
    def add_reward_stats(self, mean_reward: float, std_reward: float):
        """Add main reward statistics"""
        self.reward.append(mean_reward)
        self.reward_std.append(std_reward)
    
    def add_individual_reward(self, reward_key: str, value: float):
        """Add individual reward function value"""
        if reward_key not in self.individual_rewards:
            self.individual_rewards[reward_key] = []
        self.individual_rewards[reward_key].append(value)
    
    def clear(self):
        """Clear all metrics"""
        self.reward.clear()
        self.reward_std.clear()
        self.individual_rewards.clear()

@dataclass
class CompletionMetrics:
    """Container for completion-related metrics"""
    mean_length: List[float] = field(default_factory=list)
    min_length: List[float] = field(default_factory=list)
    max_length: List[float] = field(default_factory=list)
    clipped_ratio: List[float] = field(default_factory=list)
    mean_terminated_length: List[float] = field(default_factory=list)
    min_terminated_length: List[float] = field(default_factory=list)
    max_terminated_length: List[float] = field(default_factory=list)
    
    def add_length_stats(self, completion_lengths: List[int]):
        """Add completion length statistics"""
        self.mean_length.append(float(sum(completion_lengths)) / len(completion_lengths))
        self.min_length.append(float(min(completion_lengths)))
        self.max_length.append(float(max(completion_lengths)))
    
    def add_termination_stats(self, term_lengths: List[int], total_completions: int):
        """Add termination statistics"""
        clipped_ratio = 1 - len(term_lengths) / total_completions
        self.clipped_ratio.append(clipped_ratio)
        
        if len(term_lengths) == 0:
            term_lengths = [0]
        self.mean_terminated_length.append(float(sum(term_lengths)) / len(term_lengths))
        self.min_terminated_length.append(float(min(term_lengths)))
        self.max_terminated_length.append(float(max(term_lengths)))
    
    def clear(self):
        """Clear all metrics"""
        self.mean_length.clear()
        self.min_length.clear()
        self.max_length.clear()
        self.clipped_ratio.clear()
        self.mean_terminated_length.clear()
        self.min_terminated_length.clear()
        self.max_terminated_length.clear()

@dataclass
class ClippingMetrics:
    """Container for clipping-related metrics"""
    low_clip_mean: List[float] = field(default_factory=list)
    high_clip_mean: List[float] = field(default_factory=list)
    low_clip_min: List[float] = field(default_factory=list)
    high_clip_max: List[float] = field(default_factory=list)
    clip_ratio: List[float] = field(default_factory=list)
    
    def add_clipping_stats(self, stats: Dict[str, torch.Tensor], accelerator: Any):
        """Add clipping statistics from RL algorithm result"""
        for stat_name, stat_value in stats.items():
            gathered_stat = accelerator.gather_for_metrics(stat_value)
            if hasattr(gathered_stat, 'nanmean'):
                # Handle tensor case
                if stat_name in ['low_clip', 'high_clip']:
                    mean_val = gathered_stat.nanmean().item()
                    if stat_name == 'low_clip':
                        self.low_clip_mean.append(mean_val)
                        self.low_clip_min.append(nanmin(gathered_stat).item())
                    else:
                        self.high_clip_mean.append(mean_val)
                        self.high_clip_max.append(nanmax(gathered_stat).item())
                else:
                    self.clip_ratio.append(gathered_stat.nanmean().item())
            else:
                # Handle list/dict case - convert to tensor first
                if isinstance(gathered_stat, (list, dict)):
                    tensor_stat = torch.tensor(list(gathered_stat.values()) if isinstance(gathered_stat, dict) else gathered_stat)
                    if stat_name in ['low_clip', 'high_clip']:
                        mean_val = tensor_stat.nanmean().item()
                        if stat_name == 'low_clip':
                            self.low_clip_mean.append(mean_val)
                            self.low_clip_min.append(nanmin(tensor_stat).item())
                        else:
                            self.high_clip_mean.append(mean_val)
                            self.high_clip_max.append(nanmax(tensor_stat).item())
                    else:
                        self.clip_ratio.append(tensor_stat.nanmean().item())
    
    def clear(self):
        """Clear all metrics"""
        self.low_clip_mean.clear()
        self.high_clip_mean.clear()
        self.low_clip_min.clear()
        self.high_clip_max.clear()
        self.clip_ratio.clear()

@dataclass
class AlgorithmMetrics:
    """Container for algorithm-specific metrics"""
    kl: List[float] = field(default_factory=list)
    algorithm_specific: Dict[str, List[float]] = field(default_factory=dict)
    
    def add_kl_metric(self, kl_value: float):
        """Add KL divergence metric"""
        self.kl.append(kl_value)
    
    def add_algorithm_metric(self, algorithm_name: str, metric_name: str, value: float):
        """Add algorithm-specific metric"""
        key = f"{algorithm_name.lower()}_{metric_name}"
        if key not in self.algorithm_specific:
            self.algorithm_specific[key] = []
        self.algorithm_specific[key].append(value)
    
    def clear(self):
        """Clear all metrics"""
        self.kl.clear()
        self.algorithm_specific.clear()

@dataclass
class TrainingMetrics:
    """Container for all training metrics"""
    num_tokens: List[int] = field(default_factory=list)
    rewards: RewardMetrics = field(default_factory=RewardMetrics)
    completions: CompletionMetrics = field(default_factory=CompletionMetrics)
    clipping: ClippingMetrics = field(default_factory=ClippingMetrics)
    algorithm: AlgorithmMetrics = field(default_factory=AlgorithmMetrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging compatibility"""
        result = {}
        
        # Add simple metrics
        if self.num_tokens:
            result["num_tokens"] = self.num_tokens
        
        # Add reward metrics
        if self.rewards.reward:
            result["reward"] = self.rewards.reward
        if self.rewards.reward_std:
            result["reward_std"] = self.rewards.reward_std
        for key, values in self.rewards.individual_rewards.items():
            result[f"rewards/{key}"] = values
        
        # Add completion metrics
        for field_name in ["mean_length", "min_length", "max_length", "clipped_ratio", 
                          "mean_terminated_length", "min_terminated_length", "max_terminated_length"]:
            values = getattr(self.completions, field_name)
            if values:
                result[f"completions/{field_name}"] = values
        
        # Add clipping metrics
        if self.clipping.low_clip_mean:
            result["clip_ratio/low_clip_mean"] = self.clipping.low_clip_mean
        if self.clipping.high_clip_mean:
            result["clip_ratio/high_clip_mean"] = self.clipping.high_clip_mean
        if self.clipping.low_clip_min:
            result["clip_ratio/low_clip_min"] = self.clipping.low_clip_min
        if self.clipping.high_clip_max:
            result["clip_ratio/high_clip_max"] = self.clipping.high_clip_max
        if self.clipping.clip_ratio:
            result["clip_ratio/clip_ratio"] = self.clipping.clip_ratio
        
        # Add algorithm metrics
        if self.algorithm.kl:
            result["kl"] = self.algorithm.kl
        for key, values in self.algorithm.algorithm_specific.items():
            result[key] = values
        
        return result
    
    def clear(self):
        """Clear all metrics"""
        self.num_tokens.clear()
        self.rewards.clear()
        self.completions.clear()
        self.clipping.clear()
        self.algorithm.clear()

@dataclass
class TextualLogs:
    """Container for textual logs (prompts, completions, rewards for wandb)"""
    prompts: deque = field(default_factory=lambda: deque())
    completions: deque = field(default_factory=lambda: deque())
    rewards: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque()))
    
    def __init__(self, maxlen: Optional[int] = None):
        self.prompts = deque(maxlen=maxlen)
        self.completions = deque(maxlen=maxlen)
        self.rewards = defaultdict(lambda: deque(maxlen=maxlen))
    
    def add_logs(self, 
                 prompts: List[Any],
                 completions: List[Any],
                 reward_dict: Dict[str, Any]):
        """Add textual logs"""
        self.prompts.extend(prompts)
        self.completions.extend(completions)
        
        for reward_key, reward_values in reward_dict.items():
            self.rewards[reward_key].extend(
                reward_values.tolist() if isinstance(reward_values, torch.Tensor) else reward_values
            )
    
    def clear(self):
        """Clear all logs"""
        self.prompts.clear()
        self.completions.clear()
        for key in self.rewards:
            self.rewards[key].clear()
    
    def is_empty(self) -> bool:
        """Check if logs are empty"""
        return len(self.prompts) == 0

@dataclass
class ModeMetrics:
    """Container for train/eval mode metrics"""
    train: TrainingMetrics = field(default_factory=TrainingMetrics)
    eval: TrainingMetrics = field(default_factory=TrainingMetrics)
    
    def get_mode_metrics(self, mode: str) -> TrainingMetrics:
        """Get metrics for specific mode"""
        return self.train if mode == "train" else self.eval
    
    def clear_mode(self, mode: str):
        """Clear metrics for specific mode"""
        self.get_mode_metrics(mode).clear()
    
    def to_dict(self, mode: str) -> Dict[str, Any]:
        """Convert mode metrics to dict"""
        return self.get_mode_metrics(mode).to_dict()

# ============================================================================
# CONCRETE IMPLEMENTATIONS (Default Strategies)
# ============================================================================

class DefaultTrainingOrchestrator(TrainingOrchestrator):
    """Default training orchestration matching current GRPO behavior"""
    
    def __init__(self, gradient_accumulation_steps: int, num_iterations: int):
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_iterations = num_iterations
    
    def should_generate_new_batch(self, step: int, buffered_inputs: Any) -> bool:
        generate_every = self.gradient_accumulation_steps * self.num_iterations
        return step % generate_every == 0 or buffered_inputs is None
    
    def plan_generation_phase(self, step: int, async_generator: Any) -> GenerationPhase:
        generate_every = self.gradient_accumulation_steps * self.num_iterations
        batch_id_to_retrieve = step // generate_every
        target_batch_id = batch_id_to_retrieve + async_generator.num_batches_ahead
        
        return GenerationPhase(
            should_generate=True,
            batch_id_to_retrieve=batch_id_to_retrieve,
            target_batch_id=target_batch_id,
            batches_to_submit=list(range(batch_id_to_retrieve, target_batch_id + 1))
        )

class VLLMWeightSyncStrategy(WeightSyncStrategy):
    """Strategy for syncing weights to vLLM server"""
    
    def should_sync_weights(self, current_step: int, last_synced_step: int) -> bool:
        return current_step > last_synced_step
    
    def sync_weights(self, trainer: 'GRPOTrainer') -> None:
        trainer._move_model_to_vllm()

class DefaultBatchDataStrategy(BatchDataStrategy):
    """Default batch data gathering strategy"""
    
    def gather_batch_data(self, trainer: 'GRPOTrainer', batch_offset: int) -> BatchData:
        return trainer._gather_batch_data_impl(batch_offset)

class DefaultBatchProcessingStrategy(BatchProcessingStrategy):
    """Default batch processing matching current behavior"""
    
    def process_batch_result(self, 
                           trainer: 'GRPOTrainer',
                           batch_result: Any,
                           process_slice: slice) -> Dict[str, torch.Tensor]:
        return trainer._process_batch_result_impl(batch_result, process_slice)

class DefaultInputBufferStrategy(InputBufferStrategy):
    """Default input buffering with shuffling and splitting"""
    
    def buffer_inputs(self, 
                     trainer: 'GRPOTrainer',
                     processed_inputs: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        from errloom.training.grpo_trainer import shuffle_tensor_dict, split_tensor_dict
        
        # Shuffle and split for gradient accumulation
        shuffled_inputs = shuffle_tensor_dict(processed_inputs)
        return split_tensor_dict(shuffled_inputs, trainer.gradient_accumulation_steps)
    
    def get_step_inputs(self, 
                       trainer: 'GRPOTrainer',
                       step: int) -> Dict[str, torch.Tensor]:
        if trainer._buffered_inputs is None:
            raise RuntimeError("No buffered inputs available")
        return trainer._buffered_inputs[step % trainer.gradient_accumulation_steps]

# ============================================================================
# PIPELINE FACTORY
# ============================================================================

@dataclass
class TrainingPipelineFactory:
    """Factory for creating common training pipeline configurations"""
    
    @staticmethod
    def create_default(gradient_accumulation_steps: int, num_iterations: int) -> TrainingPipeline:
        """Create default pipeline matching current GRPO behavior"""
        return TrainingPipeline(
            orchestrator=DefaultTrainingOrchestrator(gradient_accumulation_steps, num_iterations),
            weight_sync_strategy=VLLMWeightSyncStrategy(),
            batch_data_strategy=DefaultBatchDataStrategy(),
            batch_processing_strategy=DefaultBatchProcessingStrategy(),
            input_buffer_strategy=DefaultInputBufferStrategy()
        )
    
    @staticmethod
    def create_curriculum_learning(
        gradient_accumulation_steps: int, 
        num_iterations: int,
        difficulty_schedule: str = "linear"
    ) -> TrainingPipeline:
        """Create curriculum learning pipeline (placeholder for future implementation)"""
        # TODO: Implement CurriculumOrchestrator and related strategies
        return TrainingPipelineFactory.create_default(gradient_accumulation_steps, num_iterations)
    
    @staticmethod
    def create_federated_learning(
        gradient_accumulation_steps: int,
        num_iterations: int,
        client_list: List[str],
        dp_epsilon: float = 1.0
    ) -> TrainingPipeline:
        """Create federated learning pipeline (placeholder for future implementation)"""
        # TODO: Implement FederatedOrchestrator and related strategies
        return TrainingPipelineFactory.create_default(gradient_accumulation_steps, num_iterations)
    
    @staticmethod
    def create_streaming_learning(
        window_size: int = 1000,
        latency_target_ms: int = 50,
        decay_factor: float = 0.99
    ) -> TrainingPipeline:
        """Create online/streaming learning pipeline (placeholder for future implementation)"""
        # TODO: Implement StreamingOrchestrator and related strategies
        # For now, return default with notes for future implementation
        return TrainingPipelineFactory.create_default(1, 1)  # Minimal accumulation for streaming
