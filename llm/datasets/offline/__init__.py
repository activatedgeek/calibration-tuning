def __setup():
    from importlib import import_module

    for n in [
        "combined",
        "mmlu_offline",
        "modiste",
        "offline",
    ]:
        import_module(f".{n}", __name__)


__setup()
