import argparse
import sys
from typing import Tuple

def get_base_parser() -> argparse.ArgumentParser:
    """
    Creates and returns a base argument parser with common errloom arguments.
    Programs using errloom can extend this parser with their own arguments.

    Returns:
        argparse.ArgumentParser: Base parser with common errloom arguments
    """
    parser = argparse.ArgumentParser(add_help=False)

    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--model", type=str, default="Qwen/Qwen3-4B", help="The name or path to the model to train.")
    model_group.add_argument('--max-tokens', type=int, default=2048, help='Maximum number of tokens for model responses')
    model_group.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature for model responses')

    # Runtime configuration
    runtime_group = parser.add_argument_group('Runtime Configuration')
    runtime_group.add_argument("--n", type=int, default=10, help="How many dataset rows to process with the loom.")
    runtime_group.add_argument('--dry', action='store_true', help='Run without making actual model calls')
    runtime_group.add_argument('--seed', type=int, help='Random seed for reproducibility')

    # Logging configuration
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

errloom_args = args