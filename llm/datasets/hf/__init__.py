def __setup():
    from importlib import import_module

    for n in [
        "anli",
        "arc",
        "boolq",
        "cb",
        "commonsense_qa",
        "copa",
        "cosmos_qa",
        "gsm8k",
        "hellaswag",
        "math_qa",
        "mmlu",
        "multirc",
        "obqa",
        "piqa",
        "sciq",
        "siqa",
        "snli",
        "story_cloze",
        "trec",
        "truthful_qa",
        "winogrande",
        "wsc",
    ]:
        import_module(f".{n}", __name__)


__setup()
