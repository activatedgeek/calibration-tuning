from .factory import create_model, save_model

__all__ = [
    "create_model",
    "save_model",
]

def __setup():
    from importlib import import_module

    for n in [
        "llama",
    ]:
        import_module(f".{n}", __name__)

__setup()