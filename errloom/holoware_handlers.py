import logging

from errloom.holophore import Holophore
from errloom.holoware import ClassSpan, ContextResetSpan, EgoSpan, SampleSpan
from errloom.tapestry import Context

logger = logging.getLogger(__name__)

# noinspection PyUnusedLocal
class HolowareHandlers:
    @classmethod
    def SampleSpan(cls, phore:Holophore, span:SampleSpan):
        sample = phore.sample(phore.rollout)
        if not sample:
            logger.error("Got a null sample from the loom.")
            return

        # If the SamplerSpan has a fence attribute, wrap the response in fence tags
        if span.goal:
            text = f"<{span.goal}>{sample}</{span.goal}>"
        else:
            text = sample

        phore.add_text(text)

    @classmethod
    def ContextResetSpan(cls, phore:Holophore, span:ContextResetSpan):
        phore.contexts.append(Context())
        phore.ego = 'system'

    @classmethod
    def EgoSpan(cls, phore:Holophore, span:EgoSpan):
        if phore.ego != span.ego:
            phore.ego = span.ego

    @classmethod
    def ClassSpan(cls, phore: Holophore, span: ClassSpan):
        ClassName = span.class_name
        Class = phore.env.get(ClassName)
        if not Class:
            from errloom.discovery import get_class
            Class = get_class(ClassName)

        if not Class:
            raise f"Class '{ClassName}' not found in environment or registry."

        if span.uuid not in phore.span_bindings:
            pass

        if span.body:
            span.body(phore)

        injection = phore.invoke__holo__(phore, span)
        if injection:
            phore.add_text(injection)
