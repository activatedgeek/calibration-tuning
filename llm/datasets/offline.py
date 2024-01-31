import os
import glob

from .registry import register_dataset


def get_offline(
    root=None,
    num_workers=8,
    use_cache=True,
    **_,
):
    from datasets import load_dataset, Features, Value

    features = Features(
        {
            "context": Value("string"),
            "target": Value("string"),
            "target_prompt": Value("string"),
            "prompt": Value("string"),
            # "source_dataset": Value("string"),
            "output": Value("string"),
            "query_label": Value("int32"),
        }
    )

    data_files = {}
    for split_name in ["train", "validation", "test"]:
        if os.path.isdir(f"{root}/{split_name}"):
            data_files[split_name] = glob.glob(f"{root}/{split_name}/*.csv")

    dataset = load_dataset(
        "csv",
        data_files=data_files,
        cache_dir=os.environ.get("HF_DATASETS_CACHE", root),
        features=features,
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    def _replace_none(x):
        return {k: "" if v is None else v for k, v in x.items()}

    train_data, val_data = [
        dataset[split].map(
            _replace_none,
            num_proc=num_workers,
        )
        for split in data_files.keys()
    ]

    return train_data, val_data, None


@register_dataset
def offline(*args, **kwargs):
    return get_offline(
        *args,
        **kwargs,
    )
