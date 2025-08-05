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
    def TextSpan(cls, holophore:Holophore, span:TextSpan):
        # We don't support reinforced plaintext because it's basically 100% opacity controlnet depth injection
        # It locks in the baseline "depth" which will never give a good latent space exploration
        # We need a span that implements its own mechanism where the text is not always the same
        # This way the entropy is forever fresh
        holophore.add_masked(span.text)

    @classmethod
    def ObjSpan(cls, holophore:Holophore, span:ObjSpan):
        for var_id in span.var_ids:
            if var_id in holophore.env:
                value = holophore.env[var_id]
                holophore.add_masked(f"<obj id={var_id}>")
                holophore.add_masked(str(value))
                holophore.add_masked("</obj>")
                holophore.add_masked("\n")

    @classmethod
    def SampleSpan(cls, holophore:Holophore, span:SampleSpan):
        # Add opening fence tag to context if fence is specified
        if span.fence:
            opening_tag = f"<{span.fence}>"
            holophore.add_masked(opening_tag)

        stop_sequences = []
        if span.fence:
            stop_sequences.append(f"</{span.fence}>")

        sample = holophore.sample(holophore.rollout, stop_sequences=stop_sequences)
        if not sample:
            logger.error("Got a null sample from the loom.")
            return

        # Handle the response based on whether we have a fence
        if span.fence:
            closing_tag = f"</{span.fence}>"

            if sample.endswith(closing_tag):
                # Model naturally included the closing tag
                text = sample
            else:
                # Model didn't include closing tag - add it
                text = f"{sample}{closing_tag}"
        else:
            # No fence specified - use raw sample
            text = sample

        # Add text that will be optimized and reinforced into the weights
        holophore.add_reinforced(text)

        # Universal data assignment: if span.id is set, bind the produced payload into env
        # IMPORTANT: store only the inner payload (without closing fence) for re-injection via ObjSpan
        try:
            if getattr(span, "id", None):
                payload = sample  # start from raw sample to avoid duplicate closing fences
                if span.fence:
                    open_tag = f"<{span.fence}>"
                    close_tag = f"</{span.fence}>"
                    # If model included fences inside sample, strip them; otherwise also strip any trailing closing fence
                    if payload.startswith(open_tag) and payload.endswith(close_tag):
                        payload = payload[len(open_tag):-len(close_tag)]
                    else:
                        # sample may or may not include the closing tag; ensure we do not keep the closing tag
                        if payload.endswith(close_tag):
                            payload = payload[: -len(close_tag)]
                        # and if it starts with the opening tag, remove it
                        if payload.startswith(open_tag):
                            payload = payload[len(open_tag):]
                holophore.env[span.id] = payload.strip()
        except Exception:
            # Avoid crashing sampling path for assignment issues; log at debug level
            logger.debug("Failed to assign sample payload to env id for span.id=%s", getattr(span, "id", None))

    @classmethod
    def ContextResetSpan(cls, holophore:Holophore, span:ContextResetSpan):
        holophore.new_context()
        holophore.change_ego("system")

    @classmethod
    def EgoSpan(cls, holophore:Holophore, span:EgoSpan):
        holophore.change_ego(span.ego)

    @classmethod
    def ClassSpan(cls, holophore: Holophore, span: ClassSpan):
        ClassName = span.class_name
        Class = holophore.env.get(ClassName)
        if not Class:
            from errloom.lib.discovery import get_class
            Class = get_class(ClassName)

        if not Class:
            raise Exception(f"Class '{ClassName}' not found in environment or registry.") # TODO a more appropriate error type maybe ?

        if span.uuid not in holophore.span_bindings:
            pass

        if hasattr(holophore.span_bindings[span.uuid], "__holo__"):
            insertion = holophore.invoke__holo__(holophore, span)
            if insertion:
                # Ensure injection is a string - convert Holophore to string if needed
                if hasattr(insertion, '__str__'):
                    s = str(insertion)
                else:
                    s = insertion

                holophore.add_masked(s)
        elif span.body:
            span.body.__call__(holophore)
        else:
            raise Exception(f"Nothing to be done for {ClassName} span.") # TODO a more appropriate error type maybe ?
