import os

from .registry import register_dataset

__all__ = [
    "get_trivia_qa",
]


def get_trivia_qa(root=None, subset=None, seed=None, tokenizer=None, **_):
    from datasets import load_dataset

    dataset = load_dataset(
        "trivia_qa", subset, cache_dir=os.environ.get("DATADIR", root)
    )

    tokenize_fn = lambda x: tokenizer(
        x["question"], padding="max_length", truncation=True
    )
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, num_proc=4)

    train_data = tokenized_dataset["train"].shuffle(seed=seed)
    val_data = tokenized_dataset["validation"].shuffle(seed=seed)
    test_data = tokenized_dataset["train"].shuffle(seed=seed)

    return train_data, val_data, test_data


@register_dataset
def trivia_qa(*args, **kwargs):
    return get_trivia_qa(*args, subset="unfiltered", **kwargs)


@register_dataset
def trivia_qa_rc(*args, **kwargs):
    return get_trivia_qa(*args, subset="rc", **kwargs)
