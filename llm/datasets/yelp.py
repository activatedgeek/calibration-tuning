import os

from .registry import register_dataset

__all__ = [
    "get_yelp",
]

__YELP_ATTRS = dict(num_classes=5)


def get_yelp_reviews(root=None, seed=None, tokenize_fn=None, **_):
    from datasets import load_dataset

    dataset = load_dataset(
        "yelp_review_full", cache_dir=os.environ.get("DATADIR", root)
    )

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    train_data = tokenized_dataset["train"].shuffle(seed=seed).select(range(1000))
    test_data = tokenized_dataset["test"].shuffle(seed=seed).select(range(1000))

    return train_data, None, test_data


@register_dataset(attrs=__YELP_ATTRS)
def yelp(*args, **kwargs):
    return get_yelp_reviews(*args, **kwargs)
