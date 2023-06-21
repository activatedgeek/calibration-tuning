import os

from .registry import register_dataset

__all__ = [
    "get_yelp",
]

__YELP_ATTRS = dict(num_classes=5)


def get_yelp_reviews(root=None, seed=None, tokenizer=None, **_):
    from datasets import load_dataset

    dataset = load_dataset(
        "yelp_review_full", cache_dir=os.environ.get("DATADIR", root)
    )

    tokenize_fn = lambda x: tokenizer(x["text"], padding="max_length", truncation=True)
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, num_proc=4)

    train_data = tokenized_dataset["train"].shuffle(seed=seed)
    test_data = tokenized_dataset["test"].shuffle(seed=seed)

    return train_data, None, test_data


@register_dataset(attrs=__YELP_ATTRS)
def yelp(*args, **kwargs):
    return get_yelp_reviews(*args, **kwargs)
