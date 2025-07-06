import logging
import os
from typing import Optional

from datasets import Dataset
from rich import box
from rich.panel import Panel
from rich.table import Table

from errloom import holoware_load
from errloom.log import cl
from errloom.attractor import Attractor
from errloom.loom import Loom
from errloom.holoware_load import HolowareLoader
from errloom.rollout import Rollout

logger = logging.getLogger(__name__)

# TODO allow multiple evaluation contexts

class HolowareLoom(Loom):
    """
    Compression environment that generates compression/decompression pairs.

    For each prompt, generates:
    1. Compression rollout: original_content → compressed_form
    2. Decompression rollout: compressed_form → decompressed_content

    Both rollouts receive the same reward based on compression quality + fidelity.
    This works within standard GRPO framework without trainer modifications.
    """

    def __init__(
        self,
        path: str,
        dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        eval_model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_concurrent: int = 64,
        **kwargs
    ):
        self.prompt_lib = HolowareLoader()
        self.holoware = holoware_load.load_holoware(path)
        self.message_type = 'completion'

        # if not self.dry_run:
        #     from openai import OpenAI
        #     self.evaluator_client = OpenAI(
        #         base_url="http://localhost:8000/v1",
        #         api_key="none"
        #     )
        #     self.evaluator_model = eval_model
        # else:
        #     self.evaluator_client = None
        #     self.evaluator_model = None

        cl.print(self.holoware.to_rich_debug())

        # Display environment configuration
        tb = Table(show_header=False, box=box.SIMPLE)
        tb.add_column("Parameter", style="cyan", width=25)
        tb.add_column("Value", style="white")

        tb.add_row("Holoware path", path)
        tb.add_row("Dataset size", str(len(dataset)))
        tb.add_row("Max concurrent", f"{max_concurrent}")
        tb.add_row("Evaluator model", eval_model)

        cl.print(tb)
        cl.print()

        attractor = Attractor(funcs=[], weights=[])
        super().__init__(
            roll_dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt="",
            attractor=attractor,
            max_concurrent=max_concurrent,
            message_type='chat',
            **kwargs
        )

    def run(self, state: Rollout):
        row = state.row

        def _sample_callback():
            return self.sample(state)

        env = {
            # "score-format":            self.evaluation_schema,
            "input":    row.get("text", row) if isinstance(row, dict) else row,
            "original": row.get("text", row) if isinstance(row, dict) else row,
            **({} if not isinstance(row, dict) else row),
            # self.score_class.__name__: self.score_class,
        }
        # critique_class.get_compact_schema(include_descriptions=True)

        holophore = self.holoware(state, _sample_callback, env=env)

        # # The last context contains the evaluation
        # raw_evaluation_str = format_conversation(holophore.contexts[-1].messages)
        # eval_json = holoware_parse.extract_json(raw_evaluation_str)
        # verif_obj = None

        # try:
        #     verif_obj = self.score_class.model_validate_json(eval_json)
        #     score = verif_obj.get_critique_score()
        # except Exception as e:
        #     logger.warning(f"Failed to parse or process evaluation JSON: {e}")
        # TODO we should auto-restart the evaluation, a new rollout, idk
        # TODO right now this will mess up our reward signals if the evaluator ever shits the bed and confuse the weights

        if self.dry:
            cl.print(Panel(
                holophore.to_rich(),
                title="[bold yellow]Dry Run: Full Conversation Flow[/]",
                border_style="yellow",
                box=box.ROUNDED,
                title_align="left"
            ))

        assert len(holophore.contexts) >= 2, "Expected at least 2 contexts in rollout, but found fewer."

        compressed = holophore.extract_fence("compress") or ""
        decompressed = holophore.extract_fence("decompress") or ""
        state.extra = {"compressed": compressed, "decompressed": decompressed, }


# def generate(self,
#              inputs: Dict[str, List[Any]] | Dataset | List[Any] | None = None,
#              client: OpenAI | None = None,
#              model: str | None = None,
#              sampling_args={},
#              max_concurrent: int | None = None,
#              score_rollouts: bool = True,
#              **kwargs: Any):
#     """
#     Overrides the base generate method to perform a full compression/decompression cycle.
#     """
#     # Support legacy signature where the first positional/keyword argument
#     # was named ``dataset``.
#     if max_concurrent is None:
#         max_concurrent = self.max_concurrent
#
#     # ------------------------------------------------------------------
#     # Normalise *inputs* to a flat list so we can iterate easily.
#     # ------------------------------------------------------------------
#
#     # Convert Dataset → list[dict]
#     # TODO this should be handled outside
#     # if inputs is None and 'dataset' in kwargs:
#     #     inputs = kwargs.pop('dataset')
#     # if isinstance(inputs, Dataset):
#     #     dataset_iter: List[Any] = list(inputs)
#     # elif isinstance(inputs, dict):
#     #     # Convert columnar dict → list of row dicts
#     #     keys = list(inputs.keys())
#     #     if not keys:
#     #         dataset_iter = []
#     #     else:
#     #         num_rows = len(inputs[keys[0]])
#     #         dataset_iter = [{k: inputs[k][i] for k in keys} for i in range(num_rows)]
#     # elif isinstance(inputs, list):
#     #     dataset_iter = inputs
#     # else:
#     #     raise TypeError(f"Unsupported inputs type: {type(inputs)}")
#
#     # Run the async main function
#     processed_results: List[EnvOutput] = self.run_processing(
#         items=dataset_iter,
#         client=client,
#         model=model,
#         sampling_args=sampling_args,
#         max_concurrent=max_concurrent,
#         **kwargs
#     )
#
#     processed_results = [r for r in processed_results if r is not None]
#
#     if not processed_results:
#         return EnvOutput(prompt=[], completion=[], reward=[], answer=[], state=[], extra={})
#
#     final_prompt, final_completion, final_reward, final_answer, final_state = [], [], [], [], []
#
#     final_extra = {}
#     extra_keys = None
#
#     for r in processed_results:
#         final_prompt.extend(r.prompt or [])
#         final_completion.extend(r.completion or [])
#         final_reward.extend(r.reward or [])
#         final_answer.extend(r.answer or [])
#         final_state.extend(r.state or [])
#
#         if r.extra:
#             if extra_keys is None:
#                 extra_keys = r.extra.keys()
#                 final_extra = {k: [] for k in extra_keys}
#
#             for key in extra_keys:
#                 if key in r.extra:
#                     final_extra[key].extend(r.extra[key] or [])
#
#     return EnvOutput(
#         prompt=final_prompt,
#         completion=final_completion,
#         reward=final_reward,
#         answer=final_answer,
#         state=final_state,
#         extra=final_extra,
#     )
