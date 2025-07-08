import inspect
import logging
import typing
from typing import Any, Optional

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from errloom.holoware import HoloSpan
from errloom.rollout import Context, Rollout
from errloom.utils.log import cl
from errloom.utils.openai_chat import extract_fence

if typing.TYPE_CHECKING:
    from errloom.loom import Loom
    
logger = logging.getLogger(__name__)

class Holophore:
    """
    Binding delegator around Loom and Rollout as a superset execution context for an holoware.
    All field access and modifications are delegated to the original loom and rollout object.
    """

    def __init__(self, loom: 'Loom', rollout: Rollout, env: Optional[dict[str, Any]] = None):
        # Store reference to original loom and rollout for delegation
        self._loom = loom
        self._rollout = rollout
        self.env = env or {}
        self.ego = "system"
        self.textbuf = []
        self.errors = 0
        
    def flush(self):
        if self.textbuf:
            # buftext <- self.textbuf
            text, self.textbuf = "".join(self.textbuf), []

            if self.context.messages and self.context.messages[-1]['role'] == self.ego:
                self.context.messages[-1]['content'] += text
            else:
                self.context.messages.append({'role': self.ego, 'content': text})

    def __getattr__(self, name):
        # Delegate attribute access to the original rollout, then loom
        if hasattr(self._rollout, name):
            return getattr(self._rollout, name)
        if hasattr(self._loom, name):
            return getattr(self._loom, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Handle our own attributes
        if name.startswith('_') or name in ['env']:
            super().__setattr__(name, value)
        # Delegate rollout attribute modifications to original rollout
        elif hasattr(self._rollout, name) and not hasattr(self._loom, name):
            setattr(self._rollout, name, value)
        else:
            super().__setattr__(name, value)

    @property
    def contexts(self) -> list[Context]:
        return self._rollout.contexts

    @property
    def loom(self) -> 'Loom':
        """Returns the original loom object."""
        return self._loom

    @property
    def rollout(self) -> Rollout:
        """For backward compatibility - returns the original rollout object."""
        return self._rollout
        
    def call_holofunc(self, target, funcname, args, kwargs, optional=True, filter_missing_arguments=True):
        """
        Walks the MRO of a class or instance to find and call a __holo__ method
        from its defining base class.
        If `filter_missing_arguments` is True, it inspects the function signature
        and only passes keyword arguments that are expected by the function.
        """
        if funcname == '__init__':
            if not isinstance(target, type):
                raise TypeError(f"Target for __init__ must be a class, not {type(target)}")

            final_kwargs = kwargs
            if filter_missing_arguments:
                sig = inspect.signature(target)
                final_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return target(*args, **final_kwargs)

        Impl = target if isinstance(target, type) else type(target)
        for Base in Impl.__mro__:
            if funcname in Base.__dict__:
                logger.debug("%s.%s", Base, funcname)
                # if args:
                #     logger.debug(PrintedText(args))
                # if kwargs:
                #     logger.debug(PrintedText(kwargs))
                _holofunc_ = getattr(Base, funcname)

                final_kwargs = kwargs
                if filter_missing_arguments:
                    sig = inspect.signature(_holofunc_)
                    final_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

                return _holofunc_(target, *args, **final_kwargs)

        if not optional:
            raise AttributeError(f"No {funcname} method found in MRO for {Impl}")

        return None

    def get_holofunc_args(self, span: HoloSpan):
        return [self, span], {}

    def extract_fence(self, tag, role="assistant") -> Optional[str]:
        for c in self.contexts:
            ret = extract_fence(c.messages, tag, role)
            if ret:
                return ret
        return None

    def to_rich(self) -> Table:
        renderables = []
        for i, ctx in enumerate(self.contexts):
            if i > 0:
                renderables.append(Rule(style="yellow"))
            renderables.append(ctx.text_rich)
        group = Group(*renderables)

        cl.print(Panel(
            group,
            title="[bold yellow]Dry Run: Full Conversation Flow[/]",
            border_style="yellow",
            box=box.ROUNDED,
            title_align="left"
        ))

        # Create a table for beautiful logging
        table = Table(box=box.MINIMAL, show_header=False, expand=True)
        table.add_column(style="bold magenta")
        table.add_column(style="white")
        # table.add_row("Evaluation:", evaluation)

        return table

    def __str__(self) -> str:
        return str(self.to_rich())

    def __rich_repr__(self):
        yield "loom", self._loom
        yield "rollout", self._rollout
        yield "contexts", len(self.contexts)

    def __repr__(self) -> str:
        from errloom.utils.log import PrintedText
        return str(PrintedText(self))
