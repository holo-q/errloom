import inspect
import logging
import typing
from typing import Any, Optional

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule

from errloom.holoware import Span
from errloom.tapestry import Context, Rollout
from errloom.utils.log import indent
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
        # self.textbuf:list[str] = []
        self.errors = 0
        self.span_bindings: dict[str, Any] = dict()
        self._newtext = ""

    def add_text(self, text: str):
        ctx = self.contexts[-1]
        if len(ctx.messages) == 0 or self.ego != ctx.messages[-1]['role']:
            ctx.add_message(self.ego, text)
        else:
            ctx.add_text(text)
        self._newtext += text


    def add_message(self, ego: str, text: str):
        self.contexts[-1].add_message(ego, text)
        self._newtext += text

    # @indent
    # def flush(self):
    #     if not self.textbuf:
    #         logger.warning("textbuf is already empty.")
    #         return
    #     
    #     # buftext <- self.textbuf
    #     text, self.textbuf = "".join(self.textbuf), []
    # 
    #     if self.context.messages and self.context.messages[-1]['role'] == self.ego:
    #         self.add_text(text)
    #     else:
    #         self.add_message(self.ego, text)

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

    def invoke__holo__(self, phore:'Holophore', span:Span) -> str:
        inst = phore.span_bindings.get(span.uuid, None)
        assert inst
        result = phore.invoke(inst, '__holo__', *phore.get_holofunc_args(span), optional=False)
        return result or ""
    
    def invoke(self, target, funcname, args, kwargs, optional=True, filter_missing_arguments=True):
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
                # logger.debug("%s.%s", Base.__name__, funcname)
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

    def get_holofunc_args(self, span: Span):
        return [self, span], {}

    def extract_fence(self, tag, role="assistant") -> Optional[str]:
        for c in self.contexts:
            ret = extract_fence(c.messages, tag, role)
            if ret:
                return ret
        return None

    def to_rich(self) -> Panel:
        renderables = []
        for i, ctx in enumerate(self.contexts):
            if i > 0:
                renderables.append(Rule(style="yellow"))
            renderables.append(ctx.text_rich)
        group = Group(*renderables)

        panel = Panel(
            group,
            title="[bold yellow]Dry Run: Full Conversation Flow[/]",
            border_style="yellow",
            box=box.ROUNDED,
            title_align="left"
        )

        return panel

    def __str__(self) -> str:
        return str(self.to_rich())

    def __rich_repr__(self):
        yield "loom", self._loom
        yield "rollout", self._rollout
        yield "contexts", len(self.contexts)

    def __repr__(self) -> str:
        from errloom.utils.log import PrintedText
        return str(PrintedText(self))
