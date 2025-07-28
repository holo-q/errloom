import argparse
import sys
import typing
from typing import Tuple
from errloom import defaults
from errloom.holoware_loom import HolowareLoom

# Standard arguments for an Errloom training script
# Or simply using errloom directly as a standalone program
# ------------------------------------------------------------

def colorize_command(command: str) -> str:
    """
    Colorize command examples with different colors for different parts.
    
    Args:
        command: The command string to colorize
        
    Returns:
        Rich markup string with appropriate colors applied
    """
    # Split command into parts
    parts = command.split()
    
    if not parts:
        return command
    
    colored_parts = []
    
    for i, part in enumerate(parts):
        if i == 0 and part == "uv":
            colored_parts.append(f"[bright_blue]{part}[/]")
        elif i == 1 and part == "run":
            colored_parts.append(f"[bright_blue]{part}[/]")
        elif i == 2 and part == "main":
            colored_parts.append(f"[cyan]{part}[/]")
        elif part.endswith(".hol"):
            colored_parts.append(f"[bright_green]{part}[/]")
        elif part in ["dry", "train", "dump"]:
            # Commands - bright cyan
            colored_parts.append(f"[bright_cyan]{part}[/]")
        elif part.startswith("--"):
            colored_parts.append(f"[yellow]{part}[/]")
        elif part.startswith("-"):
            colored_parts.append(f"[bright_yellow]{part}[/]")
        elif part.isdigit():
            colored_parts.append(f"[magenta]{part}[/]")
        elif ":" in part and "." in part:
            colored_parts.append(f"[magenta]{part}[/]")
        else:
            colored_parts.append(f"[white]{part}[/]")
    
    return " ".join(colored_parts)


def show_help():
    """Show helpful guidance when no command is specified."""
    from errloom.utils.log import log, logc
    from errloom.holoware_load import get_default_loader
    from rich.rule import Rule

    logc()
    log(Rule("[bold cyan]Errloom - Holoware Training", style="cyan"))
    log("")

    # Try to find available holoware
    try:
        available_holoware = get_default_loader().list_prompts()
        if available_holoware:
            log("[bold]Available holoware files:[/]")
            for hol in sorted(available_holoware):
                log(f"  [bright_green]{hol}[/bright_green]")
            log("")
        else:
            log("[yellow]No .hol files found in prompts/ or hol/ directories[/]")
            log("")
    except Exception as e:
        log(f"[yellow]Could not list available holoware files: {e}[/]")
        log("")

    log("[bold]Usage:[/]")
    log(f"  {colorize_command('uv run main <loom/ware> <command>')}")
    log("")

    log("[bold]Commands:[/]")
    log(f"  {colorize_command('dry')}     [dim]# Test run without training (mock completions)[/]")
    log(f"  {colorize_command('train')}   [dim]# Production training run[/]")
    log(f"  {colorize_command('dump')}    [dim]# Test run with rollout saving (no training)[/]")
    log("")

    log("[bold]Progressive testing workflow:[/]")
    log(f"  {colorize_command('uv run main qa.hol dry --n 1')}               [dim]# Phase 1: Test scaffolding (mock)[/]")
    log(f"  {colorize_command('uv run main qa.hol dump --lmstudio --n 3')}   [dim]# Phase 2: Test completions + save rollouts[/]")
    log(f"  {colorize_command('uv run main qa.hol train --lmstudio')}        [dim]# Phase 3: Local training[/]")
    log(f"  {colorize_command('uv run main qa.hol train --vllm')}            [dim]# Phase 4: Distributed training[/]")
    log("")

    log("[bold]Local testing on RTX 3090 / smaller GPUs:[/]")
    log(f"  {colorize_command('uv run main compressor.hol train --local-test --test-steps 3')}   [dim]# Local test mode (reduced memory)[/]")
    log(f"  {colorize_command('uv run main compressor.hol train --micro-test --test-steps 2')}   [dim]# Micro test mode (minimal memory)[/]")
    log(f"  {colorize_command('uv run main compressor.hol train --local-test --n 5')}           [dim]# Local test with more data samples[/]")
    log("")

    log("[bold]CPU debugging mode (no GPU required):[/]")
    log(f"  {colorize_command('uv run main compressor.hol train --cpu-mode --test-steps 2')}     [dim]# Debug training logic on CPU (very slow)[/]")
    log(f"  {colorize_command('uv run main compressor.hol train --cpu-mode --n 3')}              [dim]# CPU mode with more samples[/]")
    log("")

    log("[bold]Client options:[/]")
    log(f"  {colorize_command('--lmstudio')}        [dim]# Connect to LM Studio (localhost:1234)[/]")
    log(f"  {colorize_command('--client ip:port')}  [dim]# Connect to custom OpenAI-compatible server[/]")
    log(f"  {colorize_command('--vllm')}            [dim]# Use VLLM distributed client[/]")
    log(f"  {colorize_command('--openai')}          [dim]# Use OpenAI API (requires OPENAI_API_KEY)[/]")
    log("")

    log("[bold]Usage examples:[/]")
    log(f"  {colorize_command('uv run main qa.hol train')}                             [dim]# Full training run[/]")
    log(f"  {colorize_command('uv run main codemath.hol dry --client 192.168.1.100:8080')}  [dim]# Custom server test[/]")
    log("")

    log("[bold]Deployment options:[/]")
    log(f"  {colorize_command('uv run main -vai')}     [dim]# Deploy to VastAI[/]")
    log(f"  {colorize_command('uv run main -vaig')}    [dim]# Open deployment GUI[/]")
    log(f"  {colorize_command('uv run main -vaish')}   [dim]# Start shell on remote[/]")
    log(f"  {colorize_command('uv run main --help')}   [dim]# Show all options[/]")
    log("")
    log("[dim]For more information, see the documentation or run --help[/]")
    log(Rule(style="dim"))


def get_base_parser() -> argparse.ArgumentParser:
    """
    Creates and returns a base argument parser with common errloom arguments.
    Programs using errloom can extend this parser with their own arguments.

    Returns:
        argparse.ArgumentParser: Base parser with common errloom arguments
    """
    parser = argparse.ArgumentParser(add_help=True)

    # Positional arguments
    parser.add_argument("loom_or_ware", nargs="?", help="Loom class name or holoware file (.hol)")
    parser.add_argument("command", nargs="?", choices=["dry", "train", "dump"], help="Command to execute")

    loom_config = parser.add_argument_group('Loom Configuration')
    loom_config.add_argument("--ware", type=str, default=None, help="The name or path of an holoware to train.")
    loom_config.add_argument("--loom", type=str, default=HolowareLoom, help="The module name or path to a loom class to train.")
    loom_config.add_argument("--data", type=str, default=None, help="The dataset to use. (huggingface url) We will try to add more options in the future here")
    loom_config.add_argument("--model", type=str, default=defaults.DEFAULT_MODEL, help="The name or path to the model to train.")
    loom_config.add_argument('--temp', type=float, default=0.7, help='Sampling temperature for model responses')
    loom_config.add_argument('--max-ctx', type=int, default=None, help='Maximum number of tokens for model responses')
    loom_config.add_argument('--save', action='store_true', help='Generate rollouts into an output directory of the session.')

    # Client configuration
    client_group = parser.add_argument_group('Client Configuration')
    client_group.add_argument('--client', type=str, default=None, help='OpenAI-compatible server endpoint (ip:port or full URL)')
    client_group.add_argument('--lmstudio', action='store_true', help='Use LM Studio local server (127.0.0.1:1234)')
    client_group.add_argument('--vllm', action='store_true', help='Use VLLM client for distributed training')
    client_group.add_argument('--openai', action='store_true', help='Use OpenAI API (requires OPENAI_API_KEY)')

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

    # Local testing options
    test_group = parser.add_argument_group('Local Testing Configuration')
    test_group.add_argument('--local-test', action='store_true', help='Enable local testing mode (smaller batches, shorter sequences)')
    test_group.add_argument('--micro-test', action='store_true', help='Enable micro testing mode (minimal memory usage)')
    test_group.add_argument('--test-steps', type=int, default=5, help='Number of training steps for testing')
    test_group.add_argument('--cpu-mode', action='store_true', help='Run training on CPU (very slow but no GPU memory limits, good for debugging training locally)')

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

# Set command-based flags
if errlargs.command == "dry":
    errlargs.dry = True
    errlargs.dump = False
    # Default --n to 1 for dry runs
    if errlargs.n == 10:  # Only override if n is still the default value
        errlargs.n = 1
elif errlargs.command == "dump":
    errlargs.dry = True
    errlargs.dump = True
    # Default --n to 1 for dump runs
    if errlargs.n == 10:  # Only override if n is still the default value
        errlargs.n = 1
elif errlargs.command == "train":
    errlargs.dry = False
    errlargs.dump = False
else:
    # No command specified
    errlargs.dry = None
    errlargs.dump = None

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

def create_client_from_args(args, dry_run=None):
    """
    Create the appropriate client based on command line arguments.
    
    Client flags override dry_run to allow real completions during dry training runs.
    This enables progressive testing: scaffolding -> completions -> local training -> cloud training.
    
    Args:
        args: Parsed command line arguments
        dry_run: If True, uses MockClient only when no client flags are specified. 
                If None, uses args.dry value.
        
    Returns:
        OpenAI-compatible client instance
    """
    if dry_run is None:
        dry_run = getattr(args, 'dry', False)
    # Check for conflicting client options
    client_options = [args.lmstudio, args.vllm, args.openai, bool(args.client)]
    if sum(client_options) > 1:
        raise ValueError("Cannot specify multiple client types. Please choose one of: --client, --lmstudio, --vllm, --openai")
    
    # Client flags override dry_run - use real clients for completion testing
    
    # VLLM client
    if args.vllm:
        from errloom.interop.vllm_client import VLLMClient
        return VLLMClient()
    
    # LM Studio client  
    if args.lmstudio:
        from openai import OpenAI
        return OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio"  # LM Studio doesn't require a real API key
        )
    
    # Custom client endpoint
    if args.client:
        from openai import OpenAI
        endpoint = args.client
        
        # Add http:// if no protocol specified
        if not endpoint.startswith(('http://', 'https://')):
            endpoint = f"http://{endpoint}"
        
        # Add /v1 if it's just ip:port
        if not endpoint.endswith('/v1') and not '/v1/' in endpoint:
            endpoint = f"{endpoint}/v1"
            
        return OpenAI(
            base_url=endpoint,
            api_key="local"  # Default API key for local servers
        )
    
    # OpenAI API
    if args.openai:
        import os
        from openai import OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required when using --openai")
        return OpenAI(api_key=api_key)
    
    # Default: MockClient when no client specified (respects dry_run)
    if dry_run:
        from errloom.interop.mock_client import MockClient
        return MockClient()
    
    # No client specified and not dry_run - default to VLLMClient for training
    from errloom.interop.vllm_client import VLLMClient
    return VLLMClient()

# Initialize tracing if enabled
if errlargs.trace:
    from errloom.utils.log import set_trace_enabled
    set_trace_enabled(True)

# sys.argv = [sys.argv[0]] # eat up arguments