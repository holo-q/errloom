import argparse
import sys
from typing import Tuple

# Standard arguments for an Errloom training script
# Or simply using errloom directly as a standalone program
# ------------------------------------------------------------

def get_base_parser() -> argparse.ArgumentParser:
    """
    Creates and returns a base argument parser with common errloom arguments.
    Programs using errloom can extend this parser with their own arguments.

    Returns:
        argparse.ArgumentParser: Base parser with common errloom arguments
    """
    parser = argparse.ArgumentParser(add_help=False)

    loom_config = parser.add_argument_group('Loom Configuration')
    loom_config.add_argument("--ware", type=str, default=None, help="The name or path of an holoware to train.") # TODO this should be only for standalone errloom main.py, not users
    loom_config.add_argument("--loom", type=str, default=None, help="The module name or path to a loom class to train.") # TODO this should be only for standalone errloom main.py, not users
    loom_config.add_argument("--model", type=str, default="Qwen/Qwen3-4B", help="The name or path to the model to train.")
    loom_config.add_argument('--temp', type=float, default=0.7, help='Sampling temperature for model responses')
    loom_config.add_argument('--max-ctx', type=int, default=None, help='Maximum number of tokens for model responses')
    loom_config.add_argument('--dry', action='store_true', help='Run without making actual model calls')

    runtime_group = parser.add_argument_group('Runtime Configuration')
    runtime_group.add_argument('--seed', type=int, help='Random seed for reproducibility')

    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument("--n", type=int, default=10, help="How many dataset rows to process with the loom.")
    train_group.add_argument("--batch", type=int, default=8, help="How many dataset rows to process with the loom.")
    train_group.add_argument("--train-warmup", type=int, default=0, help="")
    train_group.add_argument("--train-rollouts", type=int, default=300, help="")
    train_group.add_argument("--train-plateau", type=int, default=0.1, help="Specifies the reward fluctuation threshold for plateau detection")
    train_group.add_argument("--batch-sz", type=int, default=0, help="")
    train_group.add_argument("--batch-accum", type=int, default=0, help="")
    train_group.add_argument("--train-stop", type=str, default=0.1, help="Specify a Stopper class which defines rules for stopping training, such as reward plateaus. Can call holoware to RL on this.")

    log_group = parser.add_argument_group('Logging Configuration')
    log_group.add_argument('--debug', action='store_true', help='Enable debug logging (equivalent to --log-level debug)')
    log_group.add_argument('--log-level', type=str, choices=['debug', 'info', 'warning', 'error', 'critical'], default='info', help='Set the logging level')
    log_group.add_argument('--log-more', action='store_true', help='Enable more verbose logging output')

    return parser

def parse_known_args() -> Tuple[argparse.Namespace, list]:
    """
    Parse known arguments using the base parser.
    Returns parsed known args and remaining unknown args.

    Returns:
        Tuple[argparse.Namespace, list]: Tuple of (parsed_args, unknown_args)
    """
    parser = get_base_parser()
    return parser.parse_known_args()


argv = sys.argv[1:]
args = parse_known_args()
argvr = args[1]
args = args[0]
sys.argv = [sys.argv[0]] # eat up arguments

errlargs = args

# Validate that ware and loom are not both set
if args.ware is not None and args.loom is not None:
    raise ValueError("Cannot specify both --ware and --loom. Please choose one.")
