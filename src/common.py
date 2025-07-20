from typing import Callable


def first_non_null(*args):
    for value in args:
        if value is not None:
            return value
    return None

def extract_dict(obj, *names):
    d = {}
    for x in names:
        v = getattr(obj, x)
        if isinstance(v, Callable):
            d[x] = v()
        else:
            d[x] = v
    return d


return_rv_img = dict(_='plugfun', type='img')

def dev_function(dev_return=None):
    """
    Decorates the function to return a default value if the renderer is in dev mode.
    Later we may also support requests to an external server to get the value.
    Args:
        dev_return:

    Returns:

    """

    def decorator(function):
        def wrapped_call(*kargs, **kwargs):
            from src import RenderGlobals
            # TODO update for the refactor
            if RenderGlobals.renderer.is_dev:
                def get_retval(v):
                    if isinstance(v, dict) and v.get('_') == 'plugfun':
                        if v['type'] == 'img':
                            return RenderGlobals.renderer.rv.img
                        # elif v['type'] == 'self':
                        #     return get(v['plugname'])
                        # elif v['type'] == 'redirect':
                        #     return getattr(get(v['plugname']), v['funcname']).__call__(*kargs, **kwargs)
                    return v

                if isinstance(dev_return, dict):
                    return get_retval(dev_return)
                elif isinstance(dev_return, list):
                    return [get_retval(v) for v in dev_return]
                elif isinstance(dev_return, tuple):
                    return tuple([get_retval(v) for v in dev_return])
                elif callable(dev_return):
                    return get_retval(dev_return(*kargs, **kwargs))
                else:
                    return dev_return

            return function(*kargs, **kwargs)

        return wrapped_call

    return decorator
