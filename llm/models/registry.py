import logging
from functools import wraps


__all__ = [
    "register_model",
    "get_model",
    "get_model_attrs",
]


__func_map = dict()
__attr_map = dict()


def register_model(function=None, attrs=None, **d_kwargs):
    def _decorator(f):
        @wraps(f)
        def _wrapper(*args, **kwargs):
            all_kwargs = {**d_kwargs, **kwargs}
            return f(*args, **all_kwargs)

        assert (
            _wrapper.__name__ not in __func_map
        ), f'Duplicate registration for "{_wrapper.__name__}"'

        __func_map[_wrapper.__name__] = _wrapper
        __attr_map[_wrapper.__name__] = attrs
        return _wrapper

    if function:
        return _decorator(function)
    return _decorator


def get_model_fn(name):
    if name not in __func_map:
        raise ValueError(f'Model "{name}" not found.')

    return __func_map[name]


def get_model_attrs(name):
    if name not in __attr_map:
        raise ValueError(f'Model "{name}" attributes not found.')

    return __attr_map[name]


def list_models():
    return list(__func_map.keys())


def get_model(model_name, **kwargs):
    model_fn = get_model_fn(model_name)

    model = model_fn(**kwargs)

    logging.info(f'Loaded "{model_name}".')

    return model
