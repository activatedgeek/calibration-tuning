import os
import glob

from .registry import register_dataset


def get_offline(
    root=None,
    num_workers=8,
    use_cache=True,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "csv",
        data_files=dict(train=glob.glob(f"{root}/train/*.csv")),
        cache_dir=os.environ.get("HF_DATASETS_CACHE", root),
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    def _replace_none(x):
        return {k: "" if v is None else v for k, v in x.items()}

    train_data = dataset["train"].map(
        _replace_none,
        num_proc=num_workers,
    )

    return train_data, None, None


@register_dataset
def offline(*args, **kwargs):
    return get_offline(
        *args,
        **kwargs,
    )
