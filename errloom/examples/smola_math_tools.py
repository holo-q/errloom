from datasets import concatenate_datasets

import verifiers as vf
from errloom.utils import load_example_dataset
from errloom.prompts.system_prompts import MATH_SMOLA_PROMPT_TEMPLATE
from errloom.prompts.few_shots import CALCULATOR_SMOLA_FEW_SHOTS
from errloom.looms.smola_tool_loom import SmolaToolLoom

try:
    from smolagents.default_tools import PythonInterpreterTool  # type: ignore
    from errloom.tools.smolagents import CalculatorTool
except ImportError:
    raise ImportError("Please install smolagents to use SmolAgents tools.")

"""
Multi-GPU training (single node, 4 training + 4 inference) using SmolaAgents tools

CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_server.py \
    --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num-processes 4 verifiers/examples/smola_math_tools.py
"""

dataset = load_example_dataset("math", "train", n=6000)

eval_aime24 = load_example_dataset("aime2024", n=30)
eval_aime25 = load_example_dataset("aime2025", n=30)
eval_dataset = concatenate_datasets([eval_aime24, eval_aime25]).shuffle(seed=0)

# Use SmolaAgents' PythonInterpreterTool as a replacement for the python tool
python_tool = PythonInterpreterTool(authorized_imports=["math", "sympy", "numpy"])
calculator_tool = CalculatorTool()

vf_loom = SmolaToolLoom(
    train_data=dataset,
    bench_data=eval_dataset,
    system_prompt=MATH_SMOLA_PROMPT_TEMPLATE,
    few_shot=CALCULATOR_SMOLA_FEW_SHOTS,
    tools=[python_tool, calculator_tool],
    max_steps=5
)
print(vf_loom.system_prompt)

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-smola-grpo_" + model_name.split("/")[-1].lower()

args = vf.grpo_defaults(run_name=run_name)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    loom=vf_loom,
    args=args,
)
trainer.train()
