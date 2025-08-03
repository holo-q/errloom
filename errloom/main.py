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
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
# Rich imports for beautiful output
from rich.rule import Rule

from errloom import argp, defaults, Loom
from errloom.lib import discovery
from errloom.aliases import Data
from errloom.argp import create_client_from_args, errlargs, show_help
from errloom.holoware.holoware_loom import HolowareLoom
from errloom.session import Session
from errloom.lib.log import (colorize_client, colorize_completion, colorize_deployment, colorize_error, colorize_field_label, colorize_mode_dry, colorize_mode_dump, colorize_mode_production, colorize_model, colorize_rule_title, colorize_session, colorize_target, colorize_title, log, logc, logger_main)

# discovery.crawl_package("thauten", [CommModel])
np.set_printoptions(threshold=5)

def execute_dry_run(n: int):
    client = create_client_from_args(errlargs, dry_run=True)
    HolowareLoom("compressor.hol", client=client, data="agentlans/wikipedia-paragraphs", dry=True, unsafe=errlargs.unsafe).weave(errlargs.n)
    log(Rule(colorize_mode_dry("DRY RUN COMPLETE")))

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
         default_loom: type | Loom = discovery.get_class(errlargs.loom) or HolowareLoom,  # TODO can we specify to the type checker that it's a Loom type in particular, not just any type?
         default_holoware: Optional[str] = errlargs.ware,
         default_session: Optional[Session] = None,
         parse_args: bool = True):

    if parse_args:
        argp.parse_args()

    # Early detection and routing based on arguments
    loom_or_ware_arg = getattr(errlargs, 'loom_or_ware', None)
    command_arg = getattr(errlargs, 'command', None)
    holoware_specified = default_holoware or errlargs.ware or loom_or_ware_arg

    # Check for deployment/remote actions first
    if _is_deployment_mode():
        _handle_deployment()
        return

    # If no loom/ware specified, show help
    if not loom_or_ware_arg and not holoware_specified:
        show_help()
        return

    # If no command specified, show help
    if not command_arg:
        show_help()
        return

    # Handle cat command early - no need for loom initialization
    if command_arg == "cat":
        _handle_cat_command(loom_or_ware_arg, default_holoware, type(default_loom) if default_loom else None)
        return

    # Handle loom_or_ware argument context detection
    if loom_or_ware_arg:
        # Detect if it's a holoware file (.hol extension) or loom class
        if loom_or_ware_arg.endswith('.hol') or loom_or_ware_arg in ['qa', 'tool', 'codemath', 'doublecheck', 'smola']:
            # It's a holoware file
            default_holoware = loom_or_ware_arg if loom_or_ware_arg.endswith('.hol') else f"{loom_or_ware_arg}.hol"
            default_loom = HolowareLoom
        else:
            # It's likely a loom class
            loom_class = discovery.get_class(loom_or_ware_arg)
            if loom_class:
                default_loom = loom_class
            else:
                log(colorize_error(f"❌ Unknown loom class: {loom_or_ware_arg}"))
                show_help()
                return

    setup_async()
    model_name = default_model
    LoomClass = default_loom if isinstance(default_loom, type) else type(default_loom)
    loom = default_loom if isinstance(default_loom, Loom) else None
    loom_name = loom.__class__.__name__ if loom else "HolowareLoom"

    name = default_holoware or LoomClass.__name__
    name_ext = f"{name.split('/')[-1].lower()}-{model_name.split('/')[-1].lower()}"
    title = default_title or name_ext
    session = default_session or Session.Create(title)

    # ----------------------------------------
    # logc()
    argp.print_errloom_banner()
    log(f"╔══════════════════════════════════════════════════════════════════════════════════╗")
    log(f"║                                                                                  ║")
    log(f"║                        {colorize_title('🚀 TRAINING SESSION 🚀')}                                    ║")
    log(f"║                                                                                  ║")
    log(f"╚══════════════════════════════════════════════════════════════════════════════════╝")
    log(f"")

    loom = None
    try:
        client = create_client_from_args(errlargs)
        client_type = type(client).__name__


        # Session info header
        name = default_holoware or LoomClass.__name__
        log(f"{colorize_field_label('📋 Session:')} {colorize_session(title)}")
        log(f"{colorize_field_label('🎯 Target:')} {colorize_target(name)}")
        log(f"{colorize_field_label('🧠 Model:')} {colorize_model(model_name)}")
        log(f"{colorize_field_label('🔌 Client:')} {colorize_client(client_type)}")
        log("")

        # Mode-specific messaging
        if errlargs.dry and client_type != "MockClient":
            log(colorize_mode_dry(f"🧪 DRY MODE: Training disabled, using {client_type} for real completions"))
        elif errlargs.dry:
            log(colorize_mode_dry(f"🧪 DRY MODE: Using {client_type} for mock completions"))
        elif errlargs.dry is False:
            log(colorize_mode_production(f"🏭 PRODUCTION MODE: Using {client_type} for completions and training"))
        else:
            log(f"[dim]🔄 ACTIVE MODE: Using {client_type} for completions[/]")

        log("")
        log(Rule(colorize_rule_title(f"Initializing {loom_name}"), style="cyan"))
        # ----------------------------------------

        if loom is None:
            if isinstance(LoomClass, type) and issubclass(LoomClass, HolowareLoom):
                # Check if we have a holoware from positional or --ware
                holoware_to_use = default_holoware or errlargs.ware
                if not holoware_to_use:
                    show_help()
                    return

                loom = LoomClass(
                    holoware_to_use,
                    model=model_name, tokenizer=model_name,
                    client=client,
                    data=default_data,
                    data_split=0.5,
                    dry=errlargs.dry,
                    unsafe=errlargs.unsafe,
                    show_rollout_errors=errlargs.show_rollout_errors,
                    session=session if errlargs.dump else None,
                    dump_rollouts=errlargs.dump)
            elif issubclass(LoomClass, Loom):
                loom = LoomClass(
                    model=model_name, tokenizer=model_name,
                    client=client,
                    data=default_data, data_split=0.5,
                    dry=errlargs.dry,
                    unsafe=errlargs.unsafe,
                    show_rollout_errors=errlargs.show_rollout_errors,
                    session=session if errlargs.dump else None,
                    dump_rollouts=errlargs.dump)
            else:
                raise ValueError(f"Unknown loom class: {LoomClass}")
        else:
            log(f"Using pre-supplied loom: {loom} ...")

    except Exception as e:
        log(Rule(colorize_error("❌ Initialization Crashed"), style="red"))
        raise

    # ----------------------------------------
    log(Rule("[bold cyan]Weaving[/]", style="cyan"))

    try:
        assert loom is not None

        if errlargs.command == "train":
            # For training, use the GRPO trainer
            if loom.trainer is None:
                log(Rule(colorize_error("❌ No trainer available - training requires non-dry mode")))
                return
            log("Starting training...")
            loom.trainer.train()
            log(Rule(colorize_completion("🏆 TRAINING COMPLETED")))

        else:
            # For dry/dump, just generate rollouts
            loom.weave(errlargs.n)
            if errlargs.command == "dry":
                log(Rule(colorize_mode_dry("🧪 DRY RUN COMPLETED")))
            elif errlargs.command == "dump":
                log(Rule(colorize_mode_dump("💾 DUMP COMPLETED")))
            else:
                log(Rule(colorize_completion("✅ ALL WOVEN")))
    except Exception as e:
        log(Rule(colorize_error("❌ Training Crashed")))
        raise
    finally:
        # Save session width to persistence before exiting
        from errloom.lib.log import save_session_width_to_persistence
        save_session_width_to_persistence()


def _is_deployment_mode() -> bool:
    """Check if any deployment/remote arguments are specified."""
    return (errlargs.vastai or errlargs.vastai_gui or errlargs.vastai_quick or
            errlargs.vastai_cli or errlargs.vastai_stop or errlargs.vastai_reboot or
            errlargs.vastai_destroy or errlargs.vastai_list or errlargs.vastai_trace or
            errlargs.vastai_shell or errlargs.vastai_vllm or errlargs.vastai_mount or
            errlargs.vastai_upgrade or errlargs.vastai_install or errlargs.vastai_copy or
            errlargs.vastai_search or errlargs.vastai_redeploy or errlargs.remote)


def _print_version_info():
    """Print version information for key packages."""
    import importlib.metadata
    import platform

    # Key packages to check
    packages = [
        "torch", "transformers", "accelerate", "trl", "datasets",
        "flash-attn", "vllm", "liger-kernel", "openai", "rich"
    ]

    log(f"{colorize_field_label('🐍 Python:')} {platform.python_version()}")
    log(f"{colorize_field_label('💻 Platform:')} {platform.system()} {platform.release()}")

    versions = []
    for pkg in packages:
        try:
            # Handle package name variations
            pkg_name = pkg
            if pkg == "flash-attn":
                pkg_name = "flash_attn"
            elif pkg == "liger-kernel":
                pkg_name = "liger_kernel"

            version = importlib.metadata.version(pkg_name)
            versions.append(f"{pkg}=={version}")
        except importlib.metadata.PackageNotFoundError:
            versions.append(f"{pkg}==not installed")
        except Exception:
            versions.append(f"{pkg}==unknown")

    # Print in columns for better readability
    log(f"{colorize_field_label('📦 Packages:')} {' '.join(versions[:4])}")
    if len(versions) > 4:
        log(f"{colorize_field_label('          ')} {' '.join(versions[4:8])}")
    if len(versions) > 8:
        log(f"{colorize_field_label('          ')} {' '.join(versions[8:])}")


def _handle_cat_command(loom_or_ware_arg: str | None, default_holoware: str | None, default_loom: type | None):
    """Handle the cat command to display holoware code or loom class source."""
    from errloom.lib.log import colorize_error
    from rich.rule import Rule

    logc()
    log(Rule("[bold cyan]📄 Displaying Source Code", style="cyan"))
    log("")

    try:
        # Determine what we're displaying
        if loom_or_ware_arg and (loom_or_ware_arg.endswith('.hol') or loom_or_ware_arg in ['qa', 'tool', 'codemath', 'doublecheck', 'smola']):
            # It's a holoware file
            holoware_name = loom_or_ware_arg if loom_or_ware_arg.endswith('.hol') else f"{loom_or_ware_arg}.hol"
            _display_holoware_source(holoware_name)
        elif default_holoware:
            # Use the default holoware
            _display_holoware_source(default_holoware)
        elif loom_or_ware_arg and not loom_or_ware_arg.endswith('.hol'):
            # It's likely a loom class
            _display_loom_class_source(loom_or_ware_arg)
        elif default_loom and default_loom != HolowareLoom:
            # Use the default loom class
            loom_class_name = default_loom.__name__ if hasattr(default_loom, '__name__') else str(default_loom)
            _display_loom_class_source(loom_class_name)
        else:
            log(colorize_error("❌ No holoware or loom class specified"))
            log("[dim]Please specify a .hol file or loom class name[/]")
            return

    except Exception as e:
        log(colorize_error(f"❌ Error displaying source: {e}"))
        log("[dim]Check that the file or class exists[/]")

    log(Rule(style="dim"))


def _display_holoware_source(holoware_name: str):
    """Display the source code of a holoware file."""
    from errloom.lib.log import colorize_holoware, colorize_error, colorize_success
    from errloom.holoware.holoware_loader import get_default_loader
    from rich.syntax import Syntax

    try:
        # Try to load the holoware to get its path
        loader = get_default_loader()
        holoware_path = loader.find_holoware_path(holoware_name)

        if not holoware_path:
            log(colorize_error(f"❌ Holoware file not found: {holoware_name}"))
            log("[dim]Searched in: " + ", ".join(loader.search_paths) + "[/]")
            return

        # Read and display the file content
        with open(holoware_path, 'r', encoding='utf-8') as f:
            content = f.read()

        log(f"{colorize_holoware('📄 Holoware:')} {holoware_name}")
        log(f"[dim]Path: {holoware_path}[/]")
        log("")

        # Use syntax highlighting for .hol files
        syntax = Syntax(content, "text", theme="monokai", line_numbers=True)
        log(syntax)

        log("")
        log(colorize_success("✅ Holoware source displayed"))

    except Exception as e:
        log(colorize_error(f"❌ Error reading holoware file: {e}"))


def _display_loom_class_source(loom_class_name: str):
    """Display the source code of a loom class."""
    from errloom.lib.log import colorize_loom, colorize_error, colorize_success
    from errloom.lib.discovery import get_class
    from rich.syntax import Syntax
    import inspect
    import os

    try:
        # Get the loom class
        loom_class = get_class(loom_class_name)
        if not loom_class:
            log(colorize_error(f"❌ Loom class not found: {loom_class_name}"))
            # Show available classes
            from errloom.lib.discovery import get_all_classes
            available_classes = list(get_all_classes().keys())
            if available_classes:
                log(f"[dim]Available classes: {', '.join(sorted(available_classes))}[/]")
            return

        # Get the source file
        source_file = inspect.getfile(loom_class)
        if not source_file or not os.path.exists(source_file):
            log(colorize_error(f"❌ Source file not found for class: {loom_class_name}"))
            return

        # Read and display the file content
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        log(f"{colorize_loom('🔧 Loom Class:')} {loom_class_name}")
        log(f"[dim]File: {source_file}[/]")
        log("")

        # Use syntax highlighting for Python files
        syntax = Syntax(content, "python", theme="monokai", line_numbers=True)
        log(syntax)

        log("")
        log(colorize_success("✅ Loom class source displayed"))

    except Exception as e:
        log(colorize_error(f"❌ Error reading loom class source: {e}"))


def _handle_deployment():
    """Handle deployment/remote operations."""
    from errloom import main_deploy
    import asyncio

    logc()
    log(Rule(colorize_deployment("Errloom - Remote Deployment"), style="cyan"))
    log("")

    try:

        log("[dim]Starting deployment process...[/]")
        asyncio.run(main_deploy.main())

    except ImportError as e:
        log(colorize_error(f"❌ Deployment failed: Missing dependency - {e}"))
        log("[dim]Make sure all deployment dependencies are installed.[/]")
    except Exception as e:
        log(colorize_error(f"❌ Deployment failed: {e}"))
        log("[dim]Check the logs for more details.[/]")

    log(Rule(style="dim"))


def run():
    os.makedirs("runs", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs="?", help="Session or script")

    try:
        main()
    except KeyboardInterrupt:
        logger_main.info("\nInterrupted by user")
    except Exception as e:
        logger_main.error(f"Error: {e}")
        logger_main.error(traceback.format_exc())

        log("")
        log(Rule(colorize_error("Environment"), style="red"))
        _print_version_info()
        log("")
    finally:
        # Save session width to persistence before exiting
        from errloom.lib.log import save_session_width_to_persistence
        save_session_width_to_persistence()


if __name__ == "__main__":
    run()