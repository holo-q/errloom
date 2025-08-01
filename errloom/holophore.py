import inspect
import typing
from typing import Optional

from errloom.holoware import Span
from errloom.tapestry import Rollout
from errloom.utils import log


if typing.TYPE_CHECKING:
    from errloom.loom import Loom
    from errloom.holoware import Holoware

logger = log.getLogger(__name__)

class Holophore:
    """
    The Holophore is the "soul" of the holoware, containing the state of a single
    execution of a holoware. This includes the rollout, which contains the sequence
    of contexts and samples, and the environment, which contains any variables
    or classes that are available to the holoware.
    """

    def __init__(self, loom, rollout: Rollout, env: dict):
        self._loom = loom
        self._rollout = rollout
        self.env = env
        self.span_bindings = {}
        self.errors = 0
        self._newtext = ""
        self._ego = "system"
        self.active_holowares:list['Holoware'] = list()

    @property
    def ego(self):
        return self._ego

    def change_ego(self, ego):
        self._ego = ego

    def get_class(self, classname:str):
        Class = self.env.get(classname)
        if not Class:
            from errloom.discovery import get_class
            Class = get_class(classname)
        return Class

    def find_span(self, uid):
        for ware in self.active_holowares:
            for span in ware.spans:
                if span.uuid == uid:
                    return span
        raise ValueError(f"Span not found with uid {uid}")

    def new_context(self):
        self._rollout.new_context()

    def ensure_context(self):
        if self.context is None:
            self.new_context()

    def add_text(self, text: str):
        self.ensure_context()
        ctx = self.context
        if len(ctx.messages) == 0 or self.ego != ctx.messages[-1]['role']:
            ctx.add_message(self.ego, text)
        else:
            ctx.add_text(text)
        self._newtext += text

    def add_message(self, ego: str, text: str):
        self.ensure_context()
        self.context.add_message(ego, text)
        self._newtext += text

    def add_prompt(self, content: str):
        """Add prompt text (typically masked)."""
        self.ensure_context()
        self._rollout.add_prompt(self.role, content)
        self._newtext += content

    def add_completion(self, content: str):
        """Add completion text (typically reinforced)."""
        self._rollout.add_completion(self.role, content)
        self._newtext += content

    def add_reinforced(self, content: str):
        """Add text to reinforce (unmasked in training)."""
        self._rollout.add_reinforced(self.role, content)
        self._newtext += content

    def add_masked(self, content: str):
        """Add text to mask (ignored in training)."""
        self._rollout.add_masked(self.role, content)
        self._newtext += content

    def set_mode(self, mode):
        """Set the context mode (chat/completion)."""
        self._rollout.set_mode(mode)

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
    def contexts(self):
        return self._rollout.contexts

    @property
    def context(self):
        """Get the currently active context."""
        return self._rollout.active_context

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
        def _filter_kwargs(func, passed_kwargs):
            if not filter_missing_arguments or not passed_kwargs:
                return passed_kwargs
            sig = inspect.signature(func)
            if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                return passed_kwargs
            return {k: v for k, v in passed_kwargs.items() if k in sig.parameters}

        if funcname == '__init__':
            if not isinstance(target, type):
                raise TypeError(f"Target for __init__ must be a class, not {type(target)}")

            final_kwargs = _filter_kwargs(target, kwargs)
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

                final_kwargs = _filter_kwargs(_holofunc_, kwargs)

                return _holofunc_(target, *args, **final_kwargs)

        if not optional:
            raise AttributeError(f"No {funcname} method found in MRO for {Impl}")

        return None

    def get_holofunc_args(self, span: Span):
        return [self, span], {}

    def __str__(self) -> str:
        return str(self.to_rich())

    def __rich_repr__(self):
        yield "loom", self._loom
        yield "rollout", self._rollout
        yield "contexts", len(self.contexts)

    def __repr__(self) -> str:
        from errloom.utils.log import PrintedText
        return str(PrintedText(self))
