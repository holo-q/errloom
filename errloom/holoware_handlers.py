import logging

from errloom.holophore import Holophore
from errloom.holoware import ClassSpan, ContextResetSpan, EgoSpan
from errloom.rollout import Context

logger = logging.getLogger(__name__)

class HolowareHandler:
    @classmethod
    def SamplerSpan(cls, holophore, span):
        sample = holophore.sample(holophore.rollout)
        if sample:
            # If the SamplerSpan has a fence attribute, wrap the response in fence tags
            if span.goal:
                text = f"<{span.goal}>{sample}</{span.goal}>"
            else:
                text = sample

            if holophore.context.messages and holophore.context.messages[-1]['role'] == 'assistant':
                holophore.context.messages[-1]['content'] += '\n' + text
            else:
                holophore.context.messages.append({'role': 'assistant', 'content': text})

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

        try:
            result = None
            inst = phore.context.holofunc_targets.get(span.uuid, None)
            if inst:
                # TODO extract to _call_class_init
                result = phore.call_holofunc(inst, '__holo__', *_get_holofunc_args(span), optional=False)

            # TODO what is this doing
            if isinstance(result, list) and all(isinstance(m, dict) and 'role' in m for m in result):
                phore.context.messages.extend(result)
            elif result is not None:
                if 'holo_results' not in phore.extra:
                    phore.extra['holo_results'] = []
                phore.extra['holo_results'].append({span.class_name: result})
        except Exception as e:
            logger.error(f"Failed to execute __holo__ for {span.class_name}: {e}", exc_info=True)
