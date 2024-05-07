def __setup():
    from importlib import import_module

    for n in [
        "mmlu",
    ]:
        import_module(f".{n}", __name__)


__setup()
