import verifiers as vf
from openai import OpenAI
from datasets import load_dataset

import errloom.defaults

model_name = "Qwen/Qwen2.5-7B-Instruct" # lol
judge_prompt = "Q: {question}\nA: {answer}\nGiven: {response}\nRespond with a reward between 0.0 and 1.0."
attractor = vf.CorrectnessAttractor(client=OpenAI(base_url="http://localhost:8000"), model=model_name, judge_prompt=judge_prompt)
vf_loom = vf.SingleTurnLoom(
    dataset=load_dataset("willcb/my-dataset", data_files="train"), # HF dataset with "question" and "answer" columns
    system_prompt="You are a helpful assistant.",
    attractor=attractor
)
model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.RLTrainer(loom=vf_loom, model=model, tokenizer=tokenizer, args=errloom.defaults.grpo_defaults(name="self_reward"))
trainer.train()