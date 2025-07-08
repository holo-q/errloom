import typing

if typing.TYPE_CHECKING:
    from peft import LoraConfig
    from errloom import errlargs
    from errloom.training.grpo_config import GRPOConfig

# CONSTANTS
# ----------------------------------------

DATASET_MAX_CONCURRENT = 32

# DEFAULTS
# ----------------------------------------

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_CONCURRENT = 512

def grpo_defaults(name: str) -> 'GRPOConfig':
    from errloom.training.grpo_config import GRPOConfig
    return GRPOConfig(
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

def lora_defaults(r=8, alpha=16) -> 'LoraConfig':
    from peft import LoraConfig
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
