import os
import glob
import numpy as np
from datasets import load_dataset, Features, Value

from .registry import register_dataset


def get_mmlu_oe_offline(
    root=None,
    num_workers=8,
    use_cache=True,
    **_,
):
    features = Features(
        {
            "context": Value("string"),
            "target": Value("string"),
            "target_prompt": Value("string"),
            "prompt": Value("string"),
            "output": Value("string"),
            "query_label": Value("int32"),
        }
    )

    data_files = {}
    embeddings = {}
    for split_name in ["train", "validation", "test"]:
        if os.path.isdir(f"{root}/{split_name}"):
            data_files[split_name] = glob.glob(f"{root}/{split_name}/*.csv")

        if os.path.isfile(f"{root}/{split_name}/embedding.npy"):
            embeddings[split_name] = np.load(f"{root}/{split_name}/embedding.npy")

    dataset = load_dataset("csv", data_files=data_files, features=features)
    if not use_cache:
        dataset.cleanup_cache_files()

    def _replace_none(x):
        return {k: "" if v is None else v for k, v in x.items()}

    data_splits = {
        split: dataset[split]
        .map(
            lambda _, idx: {"embedding": embeddings[split][idx]},
            with_indices=True,
            num_proc=num_workers,
        )
        .map(
            _replace_none,
            num_proc=num_workers,
        )
        for split in data_files.keys()
    }
    data_splits = {
        split: ds.with_format("np", columns=["embedding"], output_all_columns=True)
        for split, ds in data_splits.items()
    }

    train_data = data_splits.pop("train", None)
    val_data = data_splits.pop("validation", None)
    test_data = data_splits.pop("test", None)

    return train_data, val_data, test_data


@register_dataset
def mmlu_oe_offline(*args, root=None, dataset_str=None, **kwargs):
    if len(dataset_str.split(":")) == 2:
        root = dataset_str.split(":")[1]

    return get_mmlu_oe_offline(
        *args,
        **kwargs,
        root=root,
    )
