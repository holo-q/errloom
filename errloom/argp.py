import argparse
import sys
import typing
from typing import Tuple
from errloom import defaults
from errloom.holoware_loom import HolowareLoom

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
    parser = argparse.ArgumentParser(add_help=True)

    loom_config = parser.add_argument_group('Loom Configuration')
    loom_config.add_argument("positional", nargs="?", help="Positional argument that can be either a holoware file (.hol) or loom class")
    loom_config.add_argument("--ware", type=str, default=None, help="The name or path of an holoware to train.") # TODO this should be only for standalone errloom main.py, not users
    loom_config.add_argument("--loom", type=str, default=HolowareLoom, help="The module name or path to a loom class to train.") # TODO this should be only for standalone errloom main.py, not users
    loom_config.add_argument("--data", type=str, default=None, help="The dataset to use. (huggingface url) We will try to add more options in the future here") # TODO this should be only for standalone errloom main.py, not users
    loom_config.add_argument("--model", type=str, default=defaults.DEFAULT_MODEL, help="The name or path to the model to train.")
    loom_config.add_argument('--temp', type=float, default=0.7, help='Sampling temperature for model responses')
    loom_config.add_argument('--max-ctx', type=int, default=None, help='Maximum number of tokens for model responses')
    loom_config.add_argument('--dry', action='store_true', help='Run without making actual model calls')
    loom_config.add_argument('--save', action='store_true', help='Generate rollouts into an output directory of the session.')

    runtime_group = parser.add_argument_group('Runtime Configuration')
    runtime_group.add_argument('--seed', type=int, default=None, help='RNG seed for reproducibility, affects dataset order and some other things.')
    runtime_group.add_argument('--unsafe', action='store_true', help='Disable safe invocation mode for rollouts (exceptions will crash the process)')

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
    log_group.add_argument('--trace', action='store_true', help='Enable function execution tracing with timing information')
    log_group.add_argument('--print-thread-names', action='store_true', help='Print thread names in log output')
    log_group.add_argument('--show-rollout-errors', action='store_true', help='Show rollout errors in console output (default: errors logged to file only)')

    remote_group = parser.add_argument_group('Remote Configuration')
    remote_group.add_argument("--remote", action='store_true', help="Specifies whether we are running on a remote.")

    # VastAI Deployment Options
    vastai_group = parser.add_argument_group("VastAI Deployment Options")
    vastai_group.add_argument('-vai', '--vastai', action='store_true', help="Deploy to VastAI or continue the existing deploy.")
    vastai_group.add_argument('-vaig', '--vastai_gui', action='store_true', help="Open the deployment gui.")
    vastai_group.add_argument('-vaiq', '--vastai_quick', action='store_true', help="Continue a previous deployment without any copying")
    vastai_group.add_argument('-vaicli', '--vastai_cli', action='store_true', help="CLI mode.")
    vastai_group.add_argument('-svai', '--vastai_stop', action='store_true', help="Stop the VastAI instance after running.")
    vastai_group.add_argument('-rvai', '--vastai_reboot', action='store_true', help="Reboot the VastAI instance before running.")
    vastai_group.add_argument('-dvai', '--vastai_destroy', action='store_true', help="Destroy the VastAI instance after running.")
    vastai_group.add_argument('-lsvai', '--vastai_list', action='store_true', help="List the running VastAI instances.")
    vastai_group.add_argument('-vaiu', '--vastai_upgrade', action='store_true', help="Upgrade the VastAI environment. (pip installs and dependencies)")
    vastai_group.add_argument('-vaii', '--vastai_install', action='store_true', help="Install the VastAI environment. (install dependencies)")
    vastai_group.add_argument('-vaicp', '--vastai_copy', action='store_true', help="Copy fast files")
    vastai_group.add_argument('-vais', '--vastai_search', type=str, default=None, help="Search for a VastAI server")
    vastai_group.add_argument('-vaird', '--vastai_redeploy', action='store_true', help="Delete the Errloom installation and start over. (mostly used for Errloom development)")
    vastai_group.add_argument('-vait', '--vastai_trace', action='store_true', help="Trace on the VastAI errloom execution.")
    vastai_group.add_argument('-vaish', '--vastai_shell', action='store_true', help="Only start a shell on the VastAI instance.")
    vastai_group.add_argument('-vaify', '--vastai_vllm', action='store_true', help="Only start VLLM server on the VastAI instance.")
    vastai_group.add_argument('-vaim', '--vastai_mount', action='store_true', help="Mount the VastAI instance.")
    vastai_group.add_argument('--vastai_no_download', action='store_true', help="Prevent downloading during copy step.")
    vastai_group.add_argument('--shell', action='store_true', default=None, help="Open a shell in the deployed remote.")
    vastai_group.add_argument('--local', action='store_true', help="Deploy locally. (test)")

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
argv_remaining = args[1]
errlargs = args[0]

# Validate that ware and loom are not both set
if errlargs.ware is not None and errlargs.loom is not None:
    raise ValueError("Cannot specify both --ware and --loom. Please choose one.")

# VastAI deployment detection
is_vastai = errlargs.vastai or \
            errlargs.vastai_gui or \
            errlargs.vastai_upgrade or \
            errlargs.vastai_install or \
            errlargs.vastai_redeploy or \
            errlargs.vastai_quick or \
            errlargs.vastai_copy or \
            errlargs.vastai_search or \
            errlargs.vastai_destroy or \
            errlargs.vastai_reboot or \
            errlargs.vastai_list or \
            errlargs.vastai_trace or \
            errlargs.vastai_shell or \
            errlargs.vastai_vllm

is_vastai_continue = errlargs.vastai or errlargs.vastai_quick

def get_errloom_session(load=True, *, nosubdir=False):
    """
    Get the errloom session from command line arguments.
    
    Args:
        load: Whether to load the session
        nosubdir: Whether to skip subdirectory handling
        
    Returns:
        Session object or None
    """
    from errloom.session import Session

    if errlargs.positional:
        # Check if positional argument is a session name
        s = Session(errlargs.positional, load=load)
        if not nosubdir:
            # Handle subdirectory if needed
            pass  # TODO: Add subdirectory handling if needed
        return s
    
    return None

def safe_list_remove(l, value):
    """Safely remove a value from a list."""
    if not l: 
        return
    try:
        l.remove(value)
    except:
        pass

def remove_deploy_args(oargs):
    """Remove deployment-related arguments from the argument list."""
    safe_list_remove(oargs, '--dev')
    safe_list_remove(oargs, '--run')

    # Remove all that start with vastai or vai
    for prefixed_arg in oargs[:]:  # Create a copy to avoid modification during iteration
        # Remove dashes
        arg = prefixed_arg.replace('-', '')
        if arg.startswith('vastai') or arg.startswith('vai') or arg.endswith('vai'):
            safe_list_remove(oargs, prefixed_arg)

    return oargs

# Initialize tracing if enabled
if errlargs.trace:
    from errloom.utils.log import set_trace_enabled
    set_trace_enabled(True)

# sys.argv = [sys.argv[0]] # eat up arguments