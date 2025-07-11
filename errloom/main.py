"""
Main program that allows Errloom to be used directly simply
by asking for a loom or holoware to run. The main purpose of
the Errloom library is to provide an interface into executing
a loom, which is the abstraction for weaving a tapestry and
unrolling probabilistic spools.

The functions can also be called by another script.
"""
import asyncio
import concurrent
from concurrent.futures import ThreadPoolExecutor

import logging
import os

import numpy as np
# Rich imports for beautiful output
from rich.rule import Rule

from errloom import defaults, discovery
from errloom.argp import errlargs
from errloom.comm import CommModel
from errloom.defaults import DEFAULT_MODEL
from errloom.holoware_loom import HolowareLoom
from errloom.utils.log import log, logc, LogContext, logger_main

# suppress annoying insignificant bullshit spam-and-harrass-by-default behavior
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)

discovery.crawl_package("thauten", [CommModel])
np.set_printoptions(threshold=5)

def execute_dry_run(n: int):
    HolowareLoom(
        "compressor.hol",
        data="agentlans/wikipedia-paragraphs",
        alpha=0.05,
        beta=1.5,
        dry=True,
    ).weave()

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

def main(title=None,
         default_model=None,
         default_data=None,
         default_loom=HolowareLoom):
    model_name = errlargs.model or default_model or DEFAULT_MODEL
    LoomClass = discovery.get_class(errlargs.loom) or default_loom
    name = errlargs.ware or LoomClass.__name__
    name_ext = f"{name}-{model_name.split('/')[-1].lower()}"
    title = title or name_ext

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
            loom = LoomClass("hol/compressor.hol",
                model=(model_name, model),
                tokenizer=tokenizer,
                data="agentlans/wikipedia-paragraphs",
                data_split=0.5,
                dry=errlargs.dry)
    except Exception as e:
        log(Rule("[red]❌ Training Failed", style="red"))
        log(f"[bold red]❌ An error occurred during initialization: {e}[/]")
        log(Rule(style="red"))
        raise

    # ----------------------------------------

    try:
        loom.weave()

        if errlargs.dry:
            log(Rule("[bold green]🏆 TRAINING COMPLETED", style="green"))
            log(f"[bold green]🏆 Training finished successfully![/]")
            log(Rule(style="green"))
        else:
            log(Rule("[bold green] ALL WOVEN", style="green"))
            log(Rule(style="green"))
    except Exception as e:
        log(Rule("[red]❌ Training Failed", style="red"))
        log(f"[bold red]❌ An error occurred during training: {e}[/]")
        log(Rule(style="red"))
        raise


if __name__ == "__main__":
    os.makedirs("runs", exist_ok=True)

    try:
        setup_async()
        main()
    except KeyboardInterrupt:
        logger_main.info("\nInterrupted by user")
