import verifiers as vf

import errloom.defaults
from errloom.looms.doublecheck_loom import DoubleCheckLoom
from errloom.utils import load_example_dataset

SIMPLE_PROMPT = """\
You are a helpful assistant. In each turn, think step-by-step inside <think>...</think> tags, then give your final answer inside <answer>...</answer> tags.
"""

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
dataset = load_example_dataset("math", "train", n=1000)
vf_loom = DoubleCheckLoom(
    data_train=dataset,
    system_prompt=SIMPLE_PROMPT,
    few_shot=[]
)
model, tokenizer = vf.get_model_and_tokenizer(model_name)
args = errloom.defaults.grpo_defaults(name="doublecheck-{}".format(model_name.split("/")[-1].lower()))
trainer = vf.RLTrainer(
    model=model,
    tokenizer=tokenizer,
    loom=vf_loom,
    args=args,
)
trainer.train()