import logging

from errloom.holophore import Holophore
from errloom.holoware import ClassSpan, ContextResetSpan, EgoSpan, SampleSpan
from errloom.rollout import Context

logger = logging.getLogger(__name__)

# noinspection PyUnusedLocal
class HolowareHandlers:
    @classmethod
    def SamplerSpan(cls, phore:Holophore, span:SampleSpan):
        sample = phore.sample(phore.rollout)
        if sample:
            # If the SamplerSpan has a fence attribute, wrap the response in fence tags
            if span.goal:
                text = f"<{span.goal}>{sample}</{span.goal}>"
            else:
                text = sample

            phore.add_text('assistant', text)

    @classmethod
    def ContextResetSpan(cls, phore:Holophore, span:ContextResetSpan):
        phore.contexts.append(Context())
        phore.ego = 'system'

    @classmethod
    def EgoSpan(cls, phore:Holophore, span:EgoSpan):
        if phore.ego != span.ego:
            phore.flush()
            ego = span.ego

    @classmethod
    def ClassSpan(cls, phore: Holophore, span: ClassSpan):
        ClassName = span.class_name
        Class = phore.env.get(ClassName)
        if not Class:
            from errloom.discovery import get_class
            Class = get_class(ClassName)

        if not Class:
            # This should have been caught in the init loop, but as a safeguard:
            logger.error(f"Class '{ClassName}' not found for __holo__ call.")
            return

        injection = cls.invoke_holo(phore, span)
        phore.add_text(injection)
        # try:
        # except Exception as e:
        #     logger.error(f"Failed to execute __holo__ for {span.class_name}: {e}", exc_info=True)

    @classmethod
    def invoke_holo(cls, phore:Holophore, span:SampleSpan) -> str:
        logger.info(phore.span_bindings)
        inst = phore.span_bindings.get(span.uuid, None)
        assert inst
        result = phore.invoke(inst, '__holo__', *phore.get_holofunc_args(span), optional=False)
        return result or ""