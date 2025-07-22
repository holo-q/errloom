"""
Main program that allows Errloom to be used directly simply
by asking for a loom or holoware to run. The main purpose of
the Errloom library is to provide an interface into executing
a loom, which is the abstraction for weaving a tapestry and
unrolling probabilistic spools.

The functions can also be called by another script.
"""
import argparse
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
# Rich imports for beautiful output
from rich.rule import Rule

from errloom import defaults, discovery, Loom
from errloom.aliases import Data
from errloom.argp import errlargs
from errloom.comm import CommModel
from errloom.holoware_loom import HolowareLoom
from errloom.session import Session
from errloom.utils.log import log, logc, LogContext, logger_main

# discovery.crawl_package("thauten", [CommModel])
np.set_printoptions(threshold=5)

def execute_dry_run(n: int):
    HolowareLoom("compressor.hol", data="agentlans/wikipedia-paragraphs", dry=True, unsafe=errlargs.unsafe).weave(errlargs.n)
    log(Rule("[yellow]DRY RUN COMPLETE[/]"))

def setup_async():
    def setup_executor(loop):
        if loop._default_executor is None:
            executor = ThreadPoolExecutor(max_workers=defaults.DEFAULT_MAX_CONCURRENT)
            loop.set_default_executor(executor)

    try:
        evloop = asyncio.new_event_loop()
        setup_executor(evloop)
        asyncio.set_event_loop(evloop)
        # Don't close the loop here - we need it for later execution
    except RuntimeError:
        # Jupyter notebook or existing event loop
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_running_loop()
        # noinspection PyTypeChecker
        setup_executor(loop)


def main(default_title: Optional[str] = None,
         default_model: str = errlargs.model,
         default_data: str | Data = errlargs.data or "agentlans/wikipedia-paragraphs",
         default_loom: type | Loom = discovery.get_class(errlargs.loom) or HolowareLoom, # TODO can we specify to the type checker that it's a Loom type in particular, not just any type?
         default_holoware: Optional[str] = errlargs.ware,
         default_session: Optional[Session] = None):
    
    # Early detection and routing based on arguments
    positional_arg = getattr(errlargs, 'positional', None)
    holoware_specified = default_holoware or errlargs.ware or positional_arg
    
    # Check for deployment/remote actions first
    if _is_deployment_mode():
        _handle_deployment()
        return
    
    # Handle positional argument context detection
    if positional_arg:
        # Detect if it's a holoware file (.hol extension) or loom class
        if positional_arg.endswith('.hol') or positional_arg in ['qa', 'tool', 'codemath', 'doublecheck', 'smola']:
            # It's a holoware file
            default_holoware = positional_arg if positional_arg.endswith('.hol') else f"{positional_arg}.hol"
            default_loom = HolowareLoom
        else:
            # It's likely a loom class
            loom_class = discovery.get_class(positional_arg)
            if loom_class:
                default_loom = loom_class
            else:
                log(f"[bold red]❌ Unknown loom class: {positional_arg}")
                _show_help()
                return
    
    # If no holoware specified and no other clear action, show help
    if not holoware_specified and not errlargs.dry and not errlargs.save:
        _show_help()
        return
    
    setup_async()
    is_class_loom = isinstance(default_loom, type)
    model_name = default_model
    LoomClass = default_loom if is_class_loom else type(default_loom)

    name = default_holoware or LoomClass.__name__
    name_ext = f"{name}-{model_name.split('/')[-1].lower()}"
    title = default_title or name_ext
    session = default_session or Session.Create(title)

    logc()
    log(Rule(f"[bold cyan]{title}", style="cyan"))

    # ----------------------------------------

    model, tokenizer = None, None
    loom = None
    try:
        log(f"[dim]Base model: {errlargs.model}[/]")
        log(Rule(style="dim"))
        log("")

        with LogContext("🏗️ Setting up environment...", "Environment ready"):
            if is_class_loom:
                if default_loom == HolowareLoom:
                    # Check if we have a holoware from positional or --ware
                    holoware_to_use = default_holoware or errlargs.ware
                    if not holoware_to_use:
                        _show_help()
                        return
                    
                    log(f"Initializing {LoomClass.__name__} with holoware: {holoware_to_use} ...")
                    loom = LoomClass(
                        holoware_to_use,  # path argument comes first
                        data=default_data, 
                        data_split=0.5,
                        dry=errlargs.dry, 
                        unsafe=errlargs.unsafe,
                        show_rollout_errors=errlargs.show_rollout_errors)
                else:
                    log(f"Initializing {LoomClass.__name__} ...")
                    loom = LoomClass(
                        model=model_name, tokenizer=tokenizer,
                        data=default_data, data_split=0.5,
                        dry=errlargs.dry, unsafe=errlargs.unsafe,
                        show_rollout_errors=errlargs.show_rollout_errors)
            else:
                log(f"Using pre-supplied loom: {loom} ...")

    except Exception as e:
        log(Rule("[red]❌ Training Failed", style="red"))
        log(f"[bold red]❌ An error occurred during initialization: {e}")
        log(Rule(style="red"))
        raise

    # ----------------------------------------

    try:
        assert loom is not None
        loom.weave(errlargs.n)

        if errlargs.dry:
            log(Rule("[bold green]🏆 TRAINING COMPLETED"))
            log(f"[bold green]🏆 Training finished successfully!")
            log(Rule(style="green"))
        else:
            log(Rule("[bold green] ALL WOVEN"))
            log(Rule(style="green"))
    except Exception as e:
        log(Rule("[red]❌ Training Failed"))
        log(f"[bold red]❌ An error occurred during training: {e}")
        log(Rule(style="red"))
        raise


def _is_deployment_mode() -> bool:
    """Check if any deployment/remote arguments are specified."""
    return (errlargs.vastai or errlargs.vastai_gui or errlargs.vastai_quick or 
            errlargs.vastai_cli or errlargs.vastai_stop or errlargs.vastai_reboot or 
            errlargs.vastai_destroy or errlargs.vastai_list or errlargs.vastai_trace or 
            errlargs.vastai_shell or errlargs.vastai_vllm or errlargs.vastai_mount or 
            errlargs.vastai_upgrade or errlargs.vastai_install or errlargs.vastai_copy or 
            errlargs.vastai_search or errlargs.vastai_redeploy or errlargs.remote)


def _handle_deployment():
    """Handle deployment/remote operations."""
    logc()
    log(Rule("[bold cyan]Errloom - Remote Deployment", style="cyan"))
    log("")
    
    try:
        import asyncio
        from errloom.main_deploy import main as deploy_main
        
        log("[dim]Starting deployment process...[/]")
        asyncio.run(deploy_main())
        
    except ImportError as e:
        log(f"[red]❌ Deployment failed: Missing dependency - {e}[/]")
        log("[dim]Make sure all deployment dependencies are installed.[/]")
    except Exception as e:
        log(f"[red]❌ Deployment failed: {e}[/]")
        log("[dim]Check the logs for more details.[/]")
    
    log(Rule(style="dim"))


def _show_help():
    """Show helpful guidance when no holoware is specified."""
    from errloom.holoware_load import get_default_loader
    
    logc()
    log(Rule("[bold cyan]Errloom - Holoware Training", style="cyan"))
    log("")
    
    # Try to find available holoware
    try:
        available_holoware = get_default_loader().list_prompts()
        if available_holoware:
            log("[bold]Available holoware files:[/]")
            for hol in sorted(available_holoware):
                log(f"  [cyan]{hol}[/cyan]")
            log("")
        else:
            log("[yellow]No .hol files found in prompts/ or hol/ directories[/]")
            log("")
    except Exception as e:
        log(f"[yellow]Could not list available holoware files: {e}[/]")
        log("")
    
    log("[bold]Usage examples:[/]")
    log("  [cyan]uv run main prompt.hol[/cyan]               # Run prompt.hol holoware")
    log("  [cyan]uv run main qa.hol --dry --n 1[/cyan]       # Dry run with 1 sample")
    log("")
    log("[bold]Deployment options:[/]")
    log("  [cyan]uv run main -vai[/cyan]                     # Deploy to VastAI")
    log("  [cyan]uv run main -vaig[/cyan]                    # Open deployment GUI")
    log("  [cyan]uv run main -vaish[/cyan]                   # Start shell on remote")
    log("  [cyan]uv run main -vaiq[/cyan]                    # Quick deploy (no copy)")
    log("  [cyan]uv run main --help[/cyan]                   # Show all options")
    log("")
    log("[dim]For more information, see the documentation or run --help[/]")
    log(Rule(style="dim"))


def run():
    os.makedirs("runs", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs="?", help="Session or script")

    try:
        main()
    except KeyboardInterrupt:
        logger_main.info("\nInterrupted by user")
