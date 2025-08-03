import logging

from errloom.holoware.holophore import Holophore
from errloom.holoware.holoware import ClassSpan, ContextResetSpan, EgoSpan, ObjSpan, SampleSpan, TextSpan

logger = logging.getLogger(__name__)


# noinspection PyUnusedLocal
class HolowareHandlers:
    """
    Function that match the spans and get invoked when encountered during holoware execution.
    """

    @classmethod
    def TextSpan(cls, phore:Holophore, span:TextSpan):
        # We don't support reinforced plaintext because it's basically 100% opacity controlnet depth injection
        # It locks in the baseline "depth" which will never give a good latent space exploration
        # We need a span that implements its own mechanism where the text is not always the same
        # This way the entropy is forever fresh
        phore.add_masked(span.text)

    @classmethod
    def ObjSpan(cls, phore:Holophore, span:ObjSpan):
        for var_id in span.var_ids:
            if var_id in phore.env:
                value = phore.env[var_id]
                phore.add_masked(f"<obj id={var_id}>")
                phore.add_masked(str(value))
                phore.add_masked("</obj>")
                phore.add_masked("\n")

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

        # Add text that will be optimized and reinforced into the weights
        phore.add_reinforced(text)

    @classmethod
    def ContextResetSpan(cls, phore:Holophore, span:ContextResetSpan):
        phore.new_context()
        phore.change_ego("system")

    @classmethod
    def EgoSpan(cls, phore:Holophore, span:EgoSpan):
        phore.change_ego(span.ego)

    @classmethod
    def ClassSpan(cls, phore: Holophore, span: ClassSpan):
        ClassName = span.class_name
        Class = phore.env.get(ClassName)
        if not Class:
            from errloom.lib.discovery import get_class
            Class = get_class(ClassName)

        if not Class:
            raise Exception(f"Class '{ClassName}' not found in environment or registry.") # TODO a more appropriate error type maybe ?

        if span.uuid not in phore.span_bindings:
            pass

        if hasattr(phore.span_bindings[span.uuid], "__holo__"):
            insertion = phore.invoke__holo__(phore, span)
            if insertion:
                # Ensure injection is a string - convert Holophore to string if needed
                if hasattr(insertion, '__str__'):
                    s = str(insertion)
                else:
                    s = insertion

                phore.add_masked(s)
        elif span.body:
            span.body.__call__(phore)
        else:
            raise Exception(f"Nothing to be done for {ClassName} span.") # TODO a more appropriate error type maybe ?
