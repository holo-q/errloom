import logging
import inspect
import os
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Iterable, List, Optional, Union

import rich.traceback
from rich._log_render import FormatTimeCallable
from rich.abc import RichRenderable
from rich.console import Console, RenderableType
from rich.highlighter import Highlighter
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

cl = Console()
rich.traceback.install(
    show_locals=False,           # You have False - but True is great for debugging
    word_wrap=True,             # ✓ You already have this
    extra_lines=2,              # Show more context lines around the error
    max_frames=10,              # Limit very deep stacks (default is 100)
    indent_guides=True,         # Visual indentation guides
    locals_max_length=10,       # Limit local var representation length
    locals_max_string=80,       # Limit string local var length
    locals_hide_dunder=True,    # Hide __dunder__ variables in locals
    locals_hide_sunder=True,    # Hide _private variables in locals
    suppress=[],                # You can add modules to suppress here
)

class ContextualFilter(logging.Filter):
    def render(
        self,
        *,
        record: logging.LogRecord,
        traceback: Optional[inspect.Traceback],
        message_renderable,
    ):
        """Render log for display."""
        # Original render logic from RichHandler
        path = Path(record.pathname).name
        level = self.get_level_text(record)
        time_format = None if self.formatter is None else self.formatter.datefmt
        log_time = datetime.fromtimestamp(record.created)

        # Get our custom prefix
        classname = getattr(record, "classname", None)
        if classname:
            prefix = f"{classname}.{record.funcName}"
        else:
            filename = os.path.basename(record.pathname).replace(".py", "")
            prefix = f"{filename}.{record.funcName}"

        prefix_text = Text.from_markup(f"[dim]{prefix}:[/]", style="log.level")

        # Create a grid for layout
        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(style="log.time")
        table.add_column(style="log.level", width=self.level_width)

        # New column for our prefix
        table.add_column(style="log.level")

        table.add_column(ratio=1, style="log.message", overflow="fold")

        # For multiline messages, render on a new line
        is_multiline = "\n" in message_renderable.plain

        if is_multiline:
            table.add_row(
                self.render_time(log_time, time_format),
                level,
                prefix_text,
            )
            table.add_row("", "", "", message_renderable)
        else:
            table.add_row(
                self.render_time(log_time, time_format),
                level,
                prefix_text,
                message_renderable,
            )

        if traceback:
            table.add_row("", "", "", traceback)
        return table

    def render_message(self, record: logging.LogRecord, message: str):
        """
        Renders the log message with a prefix containing the caller's context.
        """
        return Text.from_markup(message)




class CustomRichHandler(RichHandler):
    """
    A custom RichHandler that formats log messages to include the
    class and function name of the caller.
    """

    def __init__(self, *kargs, print_path=False, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.print_path = print_path

    def render_message(self, record: logging.LogRecord, message: str):
        """
        Renders the log message with a prefix containing the caller's context.
        """

        path = ""
        if self.print_path:
            classname = getattr(record, "classname", None)
            if classname:
                path = f"({classname}.{record.funcName})"
            else:
                filename = os.path.basename(record.pathname).replace(".py", "")
                path = f"({filename}.{record.funcName})"

            path = f"[dim]{path}:[/dim]"

        # For multiline messages, place them on a new line after the prefix
        msg = None
        if isinstance(record.msg, str):
            msg = record.msg
        elif isinstance(record.msg, Table):
            msg = PrintedText(record.msg).markup
        elif isinstance(record.msg, Text):
            msg = record.msg.markup
        elif isinstance(record.msg, RenderableType):
            msg = str(record.msg)
        elif isinstance(record.msg, Panel):
            msg = str(record.msg)
        else:
            # raise ValueError(f"Unsupported message type: {type(record.msg)}")
            msg = PrintedText(record.msg).markup

        assert msg is not None

        if self.print_path and "\n" in msg:
            return Text.from_markup(f"{path}\n{msg}")
        else:
            return Text.from_markup(f"{msg}")


def setup_logging(
    level: str = "DEBUG",
    print_path: bool = True
) -> None:
    """
    Setup basic logging configuration for the errloom package.

    Args:
        :param print_path:
        :param level: The logging level to use. Defaults to "DEBUG".
    """

    # Get the root logger for the package
    logger = logging.getLogger()
    logger.setLevel(level.upper())

    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a RichHandler
    if not any(isinstance(handler, CustomRichHandler) for handler in logger.handlers):
        handler = CustomRichHandler(rich_tracebacks=True, show_path=False, console=cl, markup=True, print_path=print_path)
        handler.addFilter(ContextualFilter())
        logger.addHandler(handler)

    # Prevent the logger from propagating messages to the root logger
    # logger.propagate = False


def print_prompt_completions_sample(
    prompts: list[str],
    completions: list[dict],
    rewards: dict[str, list[float]],
    step: int,
    num_samples: int = 1,  # Number of samples to display
) -> None:
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    table.add_column("Reward", style="bold cyan", justify="right")

    # Get the reward values from the dictionary
    reward_values = rewards.get("reward", [])

    # Ensure we have rewards for all prompts/completions
    if len(reward_values) < len(prompts):
        # Pad with zeros if we don't have enough rewards
        reward_values = reward_values + [0.0] * (len(prompts) - len(reward_values))

    # Only show the first num_samples samples
    samples_to_show = min(num_samples, len(prompts))

    for i in range(samples_to_show):
        prompt = prompts[i]
        completion = completions[i]
        reward = reward_values[i]

        # Format prompt (can be string or list of dicts)
        formatted_prompt = Text()
        if isinstance(prompt, str):
            formatted_prompt = Text(prompt)
        elif isinstance(prompt, list):
            # For chat format, only show the last message content (typically the user's question)
            if prompt:
                last_message = prompt[-1]
                content = last_message.get("content", "")
                formatted_prompt = Text(content, style="bright_yellow")
            else:
                formatted_prompt = Text("")
        else:
            formatted_prompt = Text(str(prompt))

        # Create a formatted Text object for completion with alternating colors based on role
        formatted_completion = Text()

        if isinstance(completion, dict):
            # Handle single message dict
            role = completion.get("role", "")
            content = completion.get("content", "")
            style = "bright_cyan" if role == "assistant" else "bright_magenta"
            formatted_completion.append(f"{role}: ", style="bold")
            formatted_completion.append(content, style=style)
        elif isinstance(completion, list):
            # Handle list of message dicts
            for i, message in enumerate(completion):
                if i > 0:
                    formatted_completion.append("\n\n")

                role = message.get("role", "")
                content = message.get("content", "")

                # Set style based on role
                style = "bright_cyan" if role == "assistant" else "bright_magenta"

                formatted_completion.append(f"{role}: ", style="bold")
                formatted_completion.append(content, style=style)
        else:
            # Fallback for string completions
            formatted_completion = Text(str(completion))

        table.add_row(formatted_prompt, formatted_completion, Text(f"{reward:.2f}"))
        if i < samples_to_show - 1:  # Don't add section after last row
            pass # table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)


def ellipse(text: str, max_length: int = 50) -> str:
    """
    Truncates a string to a maximum length and adds an ellipsis if it's too long.
    """
    if len(text) > max_length:
        return text[:max_length] + '...'
    return text


def PrintedText(renderable) -> Text:
    """
    Renders any rich renderable to a Text object that can be safely logged.
    This captures the output and converts it to ANSI text, so it can be
    included in standard logging messages without breaking formatting.
    """
    console = Console()
    with console.capture() as capture:
        console.print(renderable)
    return Text.from_ansi(capture.get())