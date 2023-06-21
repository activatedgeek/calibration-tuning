import os

from .registry import register_dataset

__all__ = [
    "get_truthful_qa",
]

__TQA_ATTRS = dict(num_classes=4)


def get_truthful_qa(root=None, instance=None, seed=None, tokenizer=None, **_):
    from datasets import load_dataset

    dataset = load_dataset(
        "truthful_qa", instance, cache_dir=os.environ.get("DATADIR", root)
    )

    tokenize_fn = lambda x: tokenizer(x["question"], padding=True, truncation=True)
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, num_proc=4)

    val_data = tokenized_dataset["validation"].shuffle(seed=seed)

    return val_data, None, val_data


@register_dataset
def truthful_qa(*args, **kwargs):
    return get_truthful_qa(*args, instance="generation", **kwargs)


@register_dataset(attrs=__TQA_ATTRS)
def truthful_qa_mcq(*args, **kwargs):
    return get_truthful_qa(*args, instance="multiple_choice", **kwargs)
