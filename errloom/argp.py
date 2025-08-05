import argparse
import logging
import sys
from typing import Tuple
from errloom import defaults
from errloom.holoware.holoware_loom import HolowareLoom
from errloom.lib import log
from errloom.storage import save_last_args, load_last_args, has_last_args

# Standard arguments for an Errloom training script
# Or simply using errloom directly as a standalone program
# ------------------------------------------------------------

logger = log.getLogger(__name__)

def colorize_command(command: str) -> str:
    """
    Colorize command examples with different colors for different parts.

    Args:
        command: The command string to colorize

    Returns:
        Rich markup string with appropriate colors applied
    """
    from errloom.lib.log import (colorize_loom, colorize_holoware, colorize_session_name,
                                 colorize_placeholder, colorize_command_name)

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
        elif part in ["dry", "train", "dump", "new", "resume", "vllm"]:
            # Commands - bright cyan
            colored_parts.append(f"[bright_cyan]{part}[/]")
        elif part.startswith("--"):
            colored_parts.append(f"[yellow]{part}[/]")
        elif part.startswith("-"):
            colored_parts.append(f"[bright_yellow]{part}[/]")
        elif part.isdigit():
            colored_parts.append(f"[magenta]{part}[/]")
        elif ":" in part and ("." in part or "localhost" in part or part.startswith("127.")):
            # IP addresses, URLs, or localhost:port
            colored_parts.append(f"[magenta]{part}[/]")
        # elif part == "<loom/holoware/session_name>":
            # Special handling for the complex placeholder
            colored_text = "<" + colorize_loom("loom").replace("[/]", "") + "/" + \
                          colorize_holoware("holoware").replace("[/]", "") + "/" + \
                          colorize_session_name("session_name").replace("[/]", "") + ">"
            colored_parts.append(colored_text)
        elif part == "<command>":
            # Special handling for command placeholder
            colored_text = "<" + colorize_command_name("command").replace("[/]", "") + ">"
            colored_parts.append(colored_text)
        elif part == "<session_name>":
            # Special handling for session_name placeholder
            colored_text = "<" + colorize_session_name("session_name").replace("[/]", "") + ">"
            colored_parts.append(colored_text)
        elif part.startswith("<") and part.endswith(">"):
            # Generic placeholders
            colored_parts.append(colorize_placeholder(part))
        elif part == "..." or part == "…":
            # Ellipsis for abbreviated commands
            colored_parts.append(f"[dim]{part}[/]")
        else:
            colored_parts.append(f"[white]{part}[/]")

    return " ".join(colored_parts)


def print_errloom_banner():
    """Print the beautiful Errloom ASCII art - Retro Computing Style."""
    from errloom.lib.log import log

    # Create the ASCII art as a single multiline string to avoid RichHandler alignment issues
    ascii_art = """
[bright_yellow]╭────────────────────────────────────────────────────────────╮
│  ▄▄▄▄▄▄  ▄▄▄▄▄▄  ▄▄▄▄▄▄  ▄       ▄▄▄▄▄▄  ▄▄▄▄▄▄  ▄▄   ▄▄   │
│  ██████  ██  ██  ██  ██  ██      ██  ██  ██  ██  ███▄▄███  │
│  ██████  ██████  ██████  ██      ██  ██  ██  ██  ████████  │
│  ██▄▄▄▄  ██▄▄██  ██▄▄██  ██      ██  ██  ██  ██  ██▄██▄██  │
│  ██████  ██  ██  ██  ██  ██████  ██████  ██████  ██▄▄▄▄██  │
╰────────────────────────────────────────────────────────────╯[/]

[italic dim]To build an intelligence fractal decompression zip bomb...[/]
"""

    log(ascii_art.strip())


def show_help():
    """Display comprehensive help for errloom commands"""
    from errloom.lib.log import log, colorize_field_label
    from errloom.holoware.holoware_loader import get_default_loader
    from rich.rule import Rule

    # logc()
    print_errloom_banner()
    log("")
    log(Rule("[bold cyan]Home Screen", style="cyan"))
    log("")

    # Try to find available holoware
    try:
        available_holoware = get_default_loader().list_prompts()
        if available_holoware:
            log("[bold bright_yellow]Available holowares:[/]")
            for hol in sorted(available_holoware):
                log(f"  [bright_green]{hol}[/]")
            log("")
        else:
            log("[yellow]No .hol files found in prompts/ or hol/ directories[/]")
            log("")
    except Exception as e:
        log(f"[yellow]Could not list available holoware files: {e}[/]")
        log("")

    # Basic Usage
    log(f"[bold cyan]Basic Usage:[/bold cyan]")
    log(f"  {colorize_command('uv run main <holoware> <command> [n] [options]')}")
    log(f"  {colorize_command('uv run main <loom_class> <command> [n] [options]')}")
    log("")

    # Commands
    log(f"[bold cyan]Commands:[/bold cyan]")
    log(f"  {colorize_command('dry')}     [dim]# Run rollouts without training (uses MockClient by default)[/]")
    log(f"  {colorize_command('train')}   [dim]# Run full training with rollouts and optimization[/]")
    log(f"  {colorize_command('dump')}    [dim]# Generate and save rollouts to project directory[/]")
    log(f"  {colorize_command('cat')}     [dim]# Display holoware code or loom class source[/]")
    log("")

    # Arguments
    log(f"[bold cyan]Positional Arguments:[/bold cyan]")
    log(f"  {colorize_field_label('holoware')}      [dim]# .hol file or shorthand (qa, tool, codemath, doublecheck, smola)[/]")
    log(f"  {colorize_field_label('loom_class')}    [dim]# Loom class name for direct loom usage[/]")
    log(f"  {colorize_field_label('command')}       [dim]# One of: dry, train, dump[/]")
    log(f"  {colorize_field_label('n')}             [dim]# Number of dataset rows to process (default: 10 for train, 1 for dry/dump)[/]")
    log("")

    # Examples section
    log(f"[bold cyan]Quick Examples:[/bold cyan]")
    log(f"  {colorize_command('uv run main qa.hol dry')}                              [dim]# Quick dry run with 1 sample[/]")
    log(f"  {colorize_command('uv run main tool.hol train 50')}                       [dim]# Train with 50 samples[/]")
    log(f"  {colorize_command('uv run main codemath.hol dry 5 --debug')}              [dim]# Debug dry run with 5 samples[/]")
    log(f"  {colorize_command('uv run main smola.hol dump 3')}                        [dim]# Generate and save 3 rollouts[/]")
    log(f"  {colorize_command('uv run main qa.hol cat')}                              [dim]# Display holoware code[/]")
    log(f"  {colorize_command('uv run main HolowareLoom cat')}                        [dim]# Display loom class source[/]")
    log("")

    # Testing Examples
    log(f"[bold cyan]Testing Examples:[/bold cyan]")
    log(f"  {colorize_command('uv run main prompt.hol train 1 --micro-test')}         [dim]# Minimal test mode[/]")
    log(f"  {colorize_command('uv run main prompt.hol train 2 --local-test')}         [dim]# Local test mode[/]")
    log(f"  {colorize_command('uv run main prompt.hol train 1 --cpu --test-steps 2')} [dim]# CPU debug mode[/]")
    log(f"  {colorize_command('uv run main compressor.hol train 1 --cpu --dry')}      [dim]# Dry training mode (no backprop)[/]")
    log("")

    # Advanced Examples
    log(f"[bold cyan]Advanced Examples:[/bold cyan]")
    log(f"  {colorize_command('uv run main qa.hol train 100 --vllm --batch 16')}      [dim]# Distributed training[/]")
    log(f"  {colorize_command('uv run main custom_loom train 50 --model llama-7b')}   [dim]# Custom loom with specific model[/]")
    log(f"  {colorize_command('uv run main tool.hol train 200 --data custom_dataset')} [dim]# Training with custom dataset[/]")
    log("")

    # Client Options
    log(f"[bold cyan]Client Options:[/bold cyan]")
    log(f"  {colorize_command('--vllm')}         [dim]# Use VLLM for distributed training[/]")
    log(f"  {colorize_command('--openai')}       [dim]# Use OpenAI API (requires OPENAI_API_KEY)[/]")
    log(f"  {colorize_command('--openrouter')}   [dim]# Use OpenRouter API (requires OPENROUTER_API_KEY)[/]")
    log(f"  {colorize_command('--lmstudio')}     [dim]# Use LM Studio local server[/]")
    log(f"  {colorize_command('--client URL')}   [dim]# Custom OpenAI-compatible endpoint[/]")
    log("")

    # Testing Options
    log(f"[bold cyan]Testing & Debug Options:[/bold cyan]")
    log(f"  {colorize_command('--cpu')}          [dim]# Run on CPU (slow but unlimited memory)[/]")
    log(f"  {colorize_command('--micro-test')}   [dim]# Minimal memory usage for testing[/]")
    log(f"  {colorize_command('--local-test')}   [dim]# Optimized for local development[/]")
    log(f"  {colorize_command('--test-steps N')} [dim]# Limit training to N steps[/]")
    log(f"  {colorize_command('--debug')}        [dim]# Enable debug logging[/]")
    log(f"  {colorize_command('--unsafe')}       [dim]# Disable safe mode (crashes on errors)[/]")
    log("")

    # Deployment
    log(f"[bold cyan]Deployment:[/bold cyan]")
    log(f"  {colorize_command('uv run main --vastai')}                               [dim]# Deploy to VastAI[/]")
    log(f"  {colorize_command('uv run main --vastai-gui')}                           [dim]# VastAI deployment GUI[/]")
    log("")

    log(f"[dim]Use {colorize_command('uv run main <command> --help')} for detailed options.[/]")
    log("")
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
    parser.add_argument("command", nargs="?", choices=["dry", "train", "dump", "cat"], help="Command to execute")
    parser.add_argument("n", nargs="?", type=int, default=10, help="How many dataset rows to process with the loom")

    loom_config = parser.add_argument_group('Loom Configuration')
    loom_config.add_argument("--ware", type=str, default=None, help="The name or path of an holoware to train.")
    loom_config.add_argument("--loom", type=str, default=HolowareLoom, help="The module name or path to a loom class to train.")
    loom_config.add_argument("--data", type=str, default=None, help="The dataset to use. (huggingface url) We will try to add more options in the future here")
    loom_config.add_argument("--model", type=str, default=defaults.DEFAULT_MODEL, help="The name or path to the model to train.")
    loom_config.add_argument('--temp', type=float, default=0.7, help='Sampling temperature for model responses')
    loom_config.add_argument('--max-ctx', type=int, default=None, help='Maximum number of tokens for model responses')
    loom_config.add_argument('--save', action='store_true', help='Generate rollouts into an output directory of the project.')

    # Client configuration
    client_group = parser.add_argument_group('Client Configuration')
    client_group.add_argument('--client', type=str, default=None, help='OpenAI-compatible server endpoint (ip:port or full URL)')
    client_group.add_argument('--lmstudio', action='store_true', help='Use LM Studio local server (127.0.0.1:1234)')
    client_group.add_argument('--vllm', action='store_true', help='Use VLLM client for distributed training')
    client_group.add_argument('--openai', action='store_true', help='Use OpenAI API (requires OPENAI_API_KEY)')
    client_group.add_argument('--openrouter', action='store_true', help='Use OpenRouter API (requires OPENROUTER_API_KEY)')

    runtime_group = parser.add_argument_group('Runtime Configuration')
    runtime_group.add_argument('--seed', type=int, default=None, help='RNG seed for reproducibility, affects dataset order and some other things.')
    runtime_group.add_argument('--unsafe', action='store_true', help='Disable safe invocation mode for rollouts (exceptions will crash the process)')
    runtime_group.add_argument('--interactive', action='store_true', help='Interactive dry/dump loop: re-weave on Enter/Space or when holoware changes; Esc/q to quit')

    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument("--batch", type=int, default=8, help="How many dataset rows to process with the loom.")
    train_group.add_argument("--train-warmup", type=int, default=0, help="")
    train_group.add_argument("--train-rollouts", type=int, default=300, help="")
    train_group.add_argument("--train-plateau", type=int, default=0.1, help="Specifies the reward fluctuation threshold for plateau detection")
    train_group.add_argument("--batch-sz", type=int, default=0, help="")
    train_group.add_argument("--batch-accum", type=int, default=0, help="")
    train_group.add_argument("--train-stop", type=str, default=0.1, help="Specify a Stopper class which defines rules for stopping training, such as reward plateaus. Can call holoware to RL on this.")
    train_group.add_argument("--dry", action='store_true', help="Enable dry training mode: run training motions with MockClient but skip backprop")

    # Local testing options
    test_group = parser.add_argument_group('Local Testing Configuration')
    test_group.add_argument('--local-test', action='store_true', help='Enable local testing mode (smaller batches, shorter sequences)')
    test_group.add_argument('--micro-test', action='store_true', help='Enable micro testing mode (minimal memory usage)')
    test_group.add_argument('--test-steps', type=int, default=5, help='Number of training steps for testing')
    test_group.add_argument('--cpu', action='store_true', help='Run training on CPU (very slow but no GPU memory limits, good for debugging training locally)')

    log_group = parser.add_argument_group('Logging Configuration')
    # --debug: optional list
    #   - bare: global DEBUG
    #   - with value: comma-separated modules at DEBUG
    log_group.add_argument(
        '--debug',
        nargs='?',
        const='',
        default=None,
        help='Enable debug logging. With value: comma-separated modules to set at DEBUG (e.g., --debug tapestry,foo)'
    )
    log_group.add_argument('--log-level', type=str, choices=['debug', 'info', 'warning', 'error', 'critical'], default='info', help='Set the base logging level')
    # --logs: appendable expressions name or name:level, comma-separated
    log_group.add_argument(
        '--logs',
        action='append',
        help='Enable specific loggers with optional level. Example: --logs tapestry:debug,foo'
    )
    log_group.add_argument('--log-more', action='store_true', help='Enable more verbose logging output')
    log_group.add_argument('--trace', action='store_true', help='Enable function execution tracing with timing information')
    log_group.add_argument('--print-threads', action='store_true', help='Print thread names in log output')
    log_group.add_argument('--print-paths', action='store_true', help='Print (module.method) prefix paths in log output')
    log_group.add_argument('--show-rollout-errors', action='store_true', help='Show rollout errors in console output (default: errors logged to file only)')
    log_group.add_argument('--reset-log-columns', action='store_true', help='Reset the persisted column width for logging')

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

def parse_known_args(args: list[str] | None = None) -> Tuple[argparse.Namespace, list]:
    """
    Parse known arguments using the base parser.
    Returns parsed known args and remaining unknown args.

    Args:
        args: Optional list of arguments to parse. If None, uses sys.argv[1:]

    Returns:
        Tuple[argparse.Namespace, list]: Tuple of (parsed_args, unknown_args)
    """
    parser = get_base_parser()
    return parser.parse_known_args(args)


def parse_args() -> Tuple[argparse.Namespace, list] | None:
    """Parse command line arguments and set global variables."""
    is_testing = any(["testslide" in arg for arg in sys.argv])
    if is_testing:
        return None

    # Don't re-parse if we're not in a main entry point
    if not should_parse_args():
        return None

    args = parse_known_args()

    return args

# Intercept __RETRY__ before any local argv variables are derived
def _maybe_apply_retry():
    try:
        _argv = sys.argv[1:]
        if "__RETRY__" in _argv:
            idx = _argv.index("__RETRY__")
            before = _argv[:idx]
            overrides = _argv[idx + 1:] if len(_argv) > idx + 1 else []
            prev = load_last_args()
            prev_list = list(prev) if prev is not None else []
            overrides_list = list(overrides) if overrides is not None else []
            merged_tail = prev_list + overrides_list
            merged = before + merged_tail
            logger.debug(f"__RETRY__ activated -> replacing token at index {idx}")
            logger.debug(f"__RETRY__ before: {before}")
            logger.debug(f"__RETRY__ last_args: {prev}")
            logger.debug(f"__RETRY__ overrides: {overrides}")
            logger.debug(f"__RETRY__ final argv[1:]: {merged}")
            sys.argv = [sys.argv[0]] + merged
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Failed processing __RETRY__: {e}")
        sys.exit(2)

# Initialize globals for command line access after potential retry rewrite
argv = sys.argv[1:]  # Raw arguments without program name (possibly rewritten)
oargs = argv[:]  # Copy for manipulation
argv_remaining = oargs[1:] if len(oargs) > 1 else []  # Remaining arguments after program name

# Only parse arguments if we're running the main errloom entry point
# This prevents conflicts with other entry points like vf-vllm
def should_parse_args():
    """Check if we should parse errloom arguments based on the entry point."""
    import os
    script_name = os.path.basename(sys.argv[0])
    return not any(p in script_name for p in ["vllm", "testslide"])


# Initialize other global variables with safe defaults
is_dev = False
is_vastai = False
is_vastai_continue = False

if should_parse_args():
    # Handle __RETRY__ before parsing: replace sys.argv[1:] with last saved args and append overrides
    try:
        if "__RETRY__" in argv:
            idx = argv.index("__RETRY__")
            before = argv[:idx]
            overrides_list = argv[idx + 1:] if len(argv) > idx + 1 else []
            prev = load_last_args()
            # Merge at position: keep tokens before __RETRY__, then replay previous args, then append overrides
            prev_list = list(prev) if prev is not None else []
            merged_tail = prev_list + overrides_list
            merged = before + merged_tail
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"__RETRY__ -> before {before} | last {prev} | overrides {overrides_list} => merged {merged}")
            else:
                logger.info(f"__RETRY__ -> {' '.join(merged)}")
            sys.argv = [sys.argv[0]] + merged
            # Update working copies
            argv = merged[:]
            oargs = argv[:]
            argv_remaining = oargs[1:] if len(oargs) > 1 else []
    except Exception as e:
        logger.error(f"Failed processing __RETRY__: {e}")
        sys.exit(2)

    args = parse_args()
    assert args is not None
    argv_remaining = args[1]
    errlargs = args[0]
    # logger.info(f"errlargs: {errlargs}")

    is_vastai_continue = errlargs.vastai or errlargs.vastai_quick
else:
    # If not parsing args, errlargs and globals remain as defaults
    parser = get_base_parser()
    errlargs = parser.parse_args([])

# Automatic implication-based args
# ----------------------------------------
if errlargs.command == "dry":
    errlargs.dry = True
    errlargs.dump = False
    # Default n to 1 for dry runs if not explicitly set
    if errlargs.n is None or errlargs.n == 10:  # Default value
        errlargs.n = 1
elif errlargs.command == "dump":
    errlargs.dry = True
    errlargs.dump = True
    # Default n to 1 for dump runs if not explicitly set
    if errlargs.n is None or errlargs.n == 10:  # Default value
        errlargs.n = 1
elif errlargs.command == "train":
    # Allow explicit --dry flag during training to override
    if not hasattr(errlargs, 'dry') or not errlargs.dry:
        errlargs.dry = False
    errlargs.dump = False
    # Keep default n value of 10 for train
    if errlargs.n is None:
        errlargs.n = 10
else:
    # No command specified
    errlargs.dry = None
    errlargs.dump = None
    # Set default n if not specified
    if errlargs.n is None:
        errlargs.n = 10

is_dev = errlargs.cpu or errlargs.local_test or errlargs.micro_test or (isinstance(errlargs.debug, str) and errlargs.debug == '') or errlargs.command == 'dry'
if is_dev:
    errlargs.unsafe = True # Unsafe mode is always better for development as it allows us to debug errors as they come up

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
    client_options = [args.lmstudio, args.vllm, args.openai, args.openrouter, bool(args.client)]
    if sum(client_options) > 1:
        raise ValueError("Cannot specify multiple client types. Please choose one of: --client, --lmstudio, --vllm, --openai, --openrouter")

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
        from openai import OpenAI
        from errloom.interop.providers import get_or_prompt_openai_api_key
        api_key = get_or_prompt_openai_api_key()
        return OpenAI(api_key=api_key)

    # OpenRouter API
    if args.openrouter:
        from openai import OpenAI
        from errloom.interop.providers import get_or_prompt_openrouter_api_key
        api_key = get_or_prompt_openrouter_api_key()
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    # Default: MockClient when no client specified (respects dry_run)
    if dry_run:
        from errloom.interop.mock_client import MockClient
        return MockClient()

    # No client specified and not dry_run - default to VLLMClient for training
    from errloom.interop.vllm_client import VLLMClient
    return VLLMClient()

# Initialize tracing if enabled
if errlargs.trace:
    from errloom.lib.log import set_trace_enabled
    set_trace_enabled(True)

# Persist the effective argv (excluding program name) for future __RETRY__ runs
try:
    # Save exactly what we executed (sys.argv[1:])
    if should_parse_args():
        effective = sys.argv[1:]
        save_last_args(effective)
except Exception as e:
    logger.error(f"Failed to save last args: {e}")

# sys.argv = [sys.argv[0]] # eat up arguments