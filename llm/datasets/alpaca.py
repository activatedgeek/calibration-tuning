import os

from .registry import register_dataset

__all__ = [
    "get_alpaca_dataset",
]


def get_alpaca_dataset(
    root=None,
    mode=None,
    seed=None,
    tokenizer=None,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "tatsu-lab/alpaca", cache_dir=os.environ.get("DATADIR", root)
    )

    if mode == "mc1":
        dataset = dataset.map(
            lambda x: tokenizer(x["text"], padding=True),
            batched=True,
            num_proc=4,
            remove_columns=["instruction", "input", "output", "text"],
        )
    else:
        raise NotImplementedError

    train_data = dataset["train"].shuffle(seed=seed)

    return train_data, None, None


@register_dataset
def alpaca(*args, **kwargs):
    return get_alpaca_dataset(
        *args,
        **kwargs,
    )
