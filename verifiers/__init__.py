from typing import Callable
FnRule = Callable[..., float]

import torch._dynamo
torch._dynamo.config.suppress_errors = True # type: ignore

from .utils.logging_utils import setup_logging, print_prompt_completions_sample
from .utils.data_utils import extract_boxed_answer, extract_hash_answer, load_example_dataset
from .utils.model_utils import get_model, get_tokenizer, get_model_and_tokenizer

from .parsers.parser import Parser
from .parsers.think_parser import ThinkParser
from .parsers.xml_parser import XMLParser

from .attractors.attractor import Attractor
from .attractors.judge_rubric import CorrectnessAttractor
from .attractors.rubric_group import RouterAttractor

from .envs.loom import Loom
from .envs.multiturn_env import MultiTurnLoom
from .envs.singleturn_env import SingleTurnLoom
from .envs.tool_env import ToolLoom
from .envs.env_group import RouterLoom

from .trainers import GRPOTrainer, GRPOConfig, grpo_defaults, lora_defaults

__version__ = "0.1.0"

# Setup default logging configuration
setup_logging()

__all__ = [
    "Parser",
    "ThinkParser",
    "XMLParser",
    "Attractor",
    "CorrectnessAttractor",
    "RouterAttractor",
    "Loom",
    "MultiTurnLoom",
    "SingleTurnEnv",
    "ToolEnv",
    "RouterLoom",
    "GRPOTrainer",
    "GRPOConfig",
    "get_model",
    "get_tokenizer",
    "get_model_and_tokenizer",
    "grpo_defaults",
    "lora_defaults",
    "extract_boxed_answer",
    "extract_hash_answer",
    "load_example_dataset",
    "setup_logging",
    "print_prompt_completions_sample",
]

