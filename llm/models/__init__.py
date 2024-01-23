from .registry import register_model, get_model, get_model_attrs, list_models
from .llm_utils import get_special_tokens, load_peft_model_from_pretrained


__all__ = [
    "register_model",
    "get_model",
    "get_model_attrs",
    "list_models",
    "get_special_tokens",
    "load_peft_model_from_pretrained",
]


def __setup():
    from importlib import import_module

    for n in [
        "llama2",
        "mistral",
        "openhermes",
    ]:
        import_module(f".{n}", __name__)


__setup()
