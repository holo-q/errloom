import os
from openai import OpenAI

import verifiers as vf
from errloom.utils.data_utils import load_example_dataset, extract_boxed_answer

dataset = load_example_dataset("gsm8k").select(range(100))

system_prompt = """
Think step-by-step inside <think>...</think> tags.

Then, give your final numerical answer inside \\boxed{{...}}.
"""


def main(num_samples: int, max_tokens: int):
    parser = vf.ThinkParser(extract_fn=extract_boxed_answer)

    def correct_answer_attraction_rule(completion, answer, **kwargs):
        response = parser.parse_answer(completion) or ''
        return 1.0 if response == answer else 0.0

    attractor = vf.Attractor(funcs=[
        correct_answer_attraction_rule,
        parser.get_format_attraction_rule()
    ], weights=[1.0, 0.2])

    vf_loom = vf.SingleTurnLoom(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        attractor=attractor,
        client=OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        client_model="gpt-4.1-nano",
        client_args={
            "max_tokens":  max_tokens,
            "temperature": 0.7,
        }
    )

    rollouts = vf_loom.test(num_samples=num_samples)
    print("--- Example ---")
    # print(f"Prompt: {rollouts.rollouts[0].prompt}")
    # print(f"Completion: {rollouts.rollouts[0].completion}")
    # print(f"Answer: {rollouts.rollouts[0].answer}")
    print(f"Reward: {rollouts.rollouts[0].gravity_loom}")
    print("--- All ---")
    print("Rewards:")
    for rollout in rollouts.rollouts:
        print('base -', rollout.gravity_loom)
        for k, v in rollout.gravity_attractor:
            print(k, '-', sum(v) / len(v))

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num-samples", "-n", type=int, default=-1)
    argparser.add_argument("--max-tokens", "-t", type=int, default=2048)
    args = argparser.parse_args()
    main(args.num_samples, args.max_tokens)
