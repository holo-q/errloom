import verifiers as vf

import errloom.defaults
from errloom.utils.data_utils import load_example_dataset, extract_boxed_answer

"""
inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/examples/gsm8k.py
"""

system_prompt = """
Think step-by-step inside <think>...</think> tags.

Then, give your final numerical answer inside \\boxed{{...}}.
"""

parser = vf.ThinkParser(extract_fn=extract_boxed_answer)

# env 1: gsm8k
def gsm8k_answer_attraction_rule(completion, answer, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response == answer else 0.0
attractor1 = vf.Attractor(funcs=[
    gsm8k_answer_attraction_rule,
    parser.get_format_attraction_rule()
], weights=[1.0, 0.2])
dataset1 = load_example_dataset("gsm8k", split="train").select(range(1000))
loom1 = vf.SingleTurnLoom(
    dataset=dataset1,
    system_prompt=system_prompt,
    parser=parser,
    attractor=attractor1,
)

# env 2: math
def math_answer_attraction_rule(completion, answer, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response == answer else 0.0
attractor2 = vf.Attractor(funcs=[
    math_answer_attraction_rule,
    parser.get_format_attraction_rule()
], weights=[1.0, 0.2])
dataset2 = load_example_dataset("math", split="train").select(range(1000))
loom2 = vf.SingleTurnLoom(
    dataset=dataset2,
    system_prompt=system_prompt,
    parser=parser,
    attractor=attractor2,
)

vf_loom = vf.RouterLoom([loom1, loom2], loom_names=["gsm8k", "math"])

model_name = "willcb/Qwen3-0.6B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-grpo_" + model_name.split("/")[-1].lower()

training_args= errloom.defaults.grpo_defaults(name=run_name)
training_args.per_device_train_batch_size=16
training_args.num_rows=16
training_args.gradient_accumulation_steps=8
training_args.num_accum=1
training_args.max_context_size=512
training_args.max_completion_length=2048
training_args.max_steps=100

trainer = vf.GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    loom=vf_loom,
    args=training_args,
)
trainer.train()