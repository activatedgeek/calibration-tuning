from .registry import register_model, get_model, get_model_attrs, list_models


__all__ = [
    "register_model",
    "get_model",
    "get_model_attrs",
    "list_models",
]


def __setup():
    from importlib import import_module

    for n in [
        "llama2",
        "llama3",
        "mlp",
        "mistral",
        "mpnet",
        "openai",
        "qwen",
    ]:
        import_module(f".{n}", __name__)


__setup()
