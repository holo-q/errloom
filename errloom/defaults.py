import typing

if typing.TYPE_CHECKING:
    from peft import LoraConfig
    from errloom.training.rl_config import RLConfig

# --- CONSTANTS
DATASET_MAX_CONCURRENT = 32


# --- DEFAULTS ---
DEFAULT_MODEL = "Qwen/Qwen3-4B"  # "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_CONCURRENT = 512

def grpo_defaults(name: str) -> 'RLConfig':
    from errloom.training.rl_config import RLConfig
    from errloom import errlargs
    return RLConfig(
        run_name=name,
        learning_rate=1e-6,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=errlargs.train_warmup,
        num_train_epochs=1,
        max_steps=errlargs.train_rollouts,
        bf16=True,
        max_grad_norm=0.001,
        num_accum=errlargs.batch_accum,
        max_context_size=errlargs.max_ctx,
        max_completion_length=2048,
        per_device_train_batch_size=8,
        num_rows=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=500,
        save_only_model=True,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to="wandb",
        output_dir=f"runs/{name}"
    )

def grpo_local_test_defaults(name: str) -> 'RLConfig':
    """Optimized config for local testing on smaller GPUs (like RTX 3090)"""
    from errloom.training.rl_config import RLConfig
    from errloom import errlargs

    # Use test steps if provided, otherwise default to 5
    max_steps = errlargs.test_steps if hasattr(errlargs, 'test_steps') else 5

    return RLConfig(
        run_name=f"{name}-local-test",
        learning_rate=1e-6,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=0,  # No warmup for testing
        num_train_epochs=1,
        max_steps=max_steps,
        bf16=True,
        max_grad_norm=0.001,
        num_accum=1,  # Minimal accumulation
        max_context_size=512,  # Much shorter context
        max_completion_length=256,  # Much shorter completion
        per_device_train_batch_size=2,  # Very small batch
        num_rows=4,  # Minimal viable rows for GRPO
        gradient_accumulation_steps=2,  # Minimal accumulation
        gradient_checkpointing=True,
        save_strategy="no",  # Don't save during testing
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to=None,  # Disable wandb for testing
        output_dir=f"runs/{name}-test",
        eval_strategy="no",  # Skip evaluation
        dataloader_num_workers=0,  # Reduce memory overhead
    )

def grpo_micro_test_defaults(name: str) -> 'RLConfig':
    """Ultra minimal config for micro testing (absolute minimum memory)"""
    from errloom.training.rl_config import RLConfig
    from errloom import errlargs

    # Use test steps if provided, otherwise default to 2
    max_steps = errlargs.test_steps if hasattr(errlargs, 'test_steps') else 2

    return RLConfig(
        run_name=f"{name}-micro-test",
        learning_rate=1e-6,
        lr_scheduler_type="constant",
        warmup_steps=0,
        num_train_epochs=1,
        max_steps=max_steps,
        bf16=True,
        max_grad_norm=0.001,
        num_accum=1,
        max_context_size=128,  # Very short context
        max_completion_length=64,   # Very short completion
        per_device_train_batch_size=1,  # Single sample
        num_rows=2,  # Absolute minimum for GRPO
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        save_strategy="no",
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to=None,
        output_dir=f"runs/{name}-micro-test",
        eval_strategy="no",
        dataloader_num_workers=0,
    )

def grpo_cpu_test_defaults(name: str) -> 'RLConfig':
    """CPU mode config for debugging training logic without GPU"""
    from errloom.training.rl_config import RLConfig
    from errloom import errlargs

    # Use test steps if provided, otherwise default to 2 for CPU
    max_steps = errlargs.test_steps if hasattr(errlargs, 'test_steps') else 2

    return RLConfig(
        run_name=f"{name}-cpu-test",
        learning_rate=1e-6,
        lr_scheduler_type="constant",
        warmup_steps=0,
        num_train_epochs=1,
        max_steps=max_steps,
        fp16=False,  # Use fp32 for CPU
        bf16=False,  # BF16 not supported on most CPUs
        max_grad_norm=0.001,
        num_accum=1,
        max_context_size=64,   # Very short for CPU
        max_completion_length=32,  # Very short for CPU
        per_device_train_batch_size=2,  # Increased to 2 so effective batch = 2
        num_rows=2,  # Minimum for GRPO, now 2 divides 2 evenly
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,  # Can cause issues on CPU
        save_strategy="no",
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to=[],  # Empty list to disable all logging
        output_dir=f"runs/{name}-cpu-test",
        eval_strategy="no",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,  # Not needed for CPU
        use_cpu=True,  # Force CPU usage
        disable_nccl_init=True,  # Disable NCCL for local testing
    )

def lora_defaults(r=8, alpha=16) -> 'LoraConfig':
    from peft import LoraConfig
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
