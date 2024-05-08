import logging
from functools import wraps


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
        __attr_map[_wrapper.__name__] = attrs or dict()
        return _wrapper

    if function:
        return _decorator(function)
    return _decorator


model_key = lambda m: m.split(":")[0]


def get_model_attrs(name):
    key = model_key(name)
    if key not in __attr_map:
        raise ValueError(f'Model "{key}" not found.')

    return __attr_map[key]


def get_model_fn(name):
    key = model_key(name)
    if key not in __func_map:
        raise ValueError(f'Model "{key}" not found.')

    return __func_map[key]


def get_model(model_name, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_fn = get_model_fn(model_name)

    model = model_fn(model_str=model_name, **kwargs)

    logging.info(f'Loaded "{model_name}".')

    return model


def list_models():
    return [
        model_name
        for model_name in __func_map.keys()
        if not get_model_attrs(model_name).get("unlisted", False)
    ]
