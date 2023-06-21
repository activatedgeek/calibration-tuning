import os

from .registry import register_dataset

__all__ = [
    "get_truthful_qa",
]


def get_truthful_qa(root=None, instance=None, seed=None, tokenizer=None, **_):
    from datasets import load_dataset

    dataset = load_dataset(
        "truthful_qa", instance, cache_dir=os.environ.get("DATADIR", root)
    )

    tokenize_fn = lambda x: tokenizer(x["question"], padding="max_length", truncation=True)
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, num_proc=4)

    val_data = tokenized_dataset["validation"].shuffle(seed=seed)

    ## FIXME: don't return train and test.
    return val_data, val_data, val_data


@register_dataset
def truthful_qa(*args, **kwargs):
    return get_truthful_qa(*args, instance="generation", **kwargs)


@register_dataset
def truthful_qa_mcq(*args, **kwargs):
    return get_truthful_qa(*args, instance="multiple_choice", **kwargs)
