from .registry import register_model, get_model, get_model_attrs, list_models
from .llm_utils import get_special_tokens


__all__ = [
    "register_model",
    "get_model",
    "get_model_attrs",
    "list_models",
    "get_special_tokens",
]


def __setup():
    from importlib import import_module

    for n in [
        "llama",
        "llama2",
    ]:
        import_module(f".{n}", __name__)


__setup()
