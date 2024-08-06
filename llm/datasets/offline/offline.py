import logging
import os
import glob
from enum import Enum
import numpy as np
from datasets import load_dataset, Features, Value, DatasetDict

from ..registry import register_dataset
from ..llm_utils_oe import sanitize_generations


CSV_DATASET_FEATURES = Features(
    {
        "context": Value("string"),
        "target": Value("string"),
        "target_prompt": Value("string"),
        "prompt": Value("string"),
        "output": Value("string"),
        "query_label": Value("int32"),
    }
)


class DatasetSizeRatio(float, Enum):
    XXS = 0.01
    XS = 0.1
    SM = 0.25
    MD = 0.5


def get_offline(
    seed=None,
    root=None,
    num_workers=8,
    use_cache=True,
    data_ratio=None,
    train_kshot=0,
    eval_kshot=0,
    load_embeddings=True,
    **_,
):
    data_files = {}
    embeddings = {}
    for split_name in ["train", "validation", "test"]:
        if os.path.isdir(f"{root}/{split_name}"):
            data_files[split_name] = glob.glob(f"{root}/{split_name}/*.csv")

            if os.path.isfile(f"{root}/{split_name}/embedding.npy"):
                embeddings[split_name] = np.load(f"{root}/{split_name}/embedding.npy")

    dataset = load_dataset("csv", data_files=data_files, features=CSV_DATASET_FEATURES)
    if not use_cache:
        dataset.cleanup_cache_files()

    dataset = dataset.map(
        lambda x: {k: "" if v is None else v for k, v in x.items()},
        num_proc=num_workers,
    )

    if load_embeddings and len(set(dataset.keys()) - set(embeddings.keys())) == 0:
        dataset = DatasetDict(
            {
                split: ds.map(
                    lambda _, idx: {"embedding": embeddings[split][idx]},
                    with_indices=True,
                    num_proc=num_workers,
                )
                for split, ds in dataset.items()
            }
        )
        dataset = dataset.with_format(
            "np", columns=["embedding"], output_all_columns=True
        )

    if data_ratio is not None:
        data_ratio = DatasetSizeRatio(data_ratio)

        def sub_sample(data):
            N = len(data)
            idxs = np.random.default_rng(seed=seed).choice(
                range(N), int(data_ratio * N)
            )
            return data.select(idxs)

        dataset = DatasetDict({split: sub_sample(ds) for split, ds in dataset.items()})

    prompt_kshot = {
        "train": train_kshot,
        "validation": eval_kshot,
        "test": eval_kshot,
    }
    data_splits = DatasetDict(
        {
            split: (
                ds.remove_columns([c for c in ["prompt"] if c in ds.column_names])
                if prompt_kshot[split] == 0
                else ds
            )
            for split, ds in dataset.items()
        }
    ).map(lambda s: {"output": sanitize_generations([s["output"].strip()])[0]})

    train_data = data_splits.pop("train", None)
    val_data = data_splits.pop("validation", None)
    test_data = data_splits.pop("test", None)

    return train_data, val_data, test_data


@register_dataset(attrs=dict(unlisted=True))
def offline(*args, root=None, dataset_str=None, prompt_style=None, **kwargs):
    try:
        _, name = dataset_str.split(":")
    except ValueError:
        logging.exception(
            f'Dataset string should be formatted as "offline:<name>" (Got {dataset_str})',
        )
        raise

    root = f"{root}/offline/{name}-{prompt_style}"

    return get_offline(*args, root=root, **kwargs)


@register_dataset(attrs=dict(unlisted=True))
def offline_xxs(*args, **kwargs):
    kwargs.pop("data_ratio", None)
    return offline(*args, data_ratio=DatasetSizeRatio.XXS, **kwargs)


@register_dataset(attrs=dict(unlisted=True))
def offline_xs(*args, **kwargs):
    kwargs.pop("data_ratio", None)
    return offline(*args, data_ratio=DatasetSizeRatio.XS, **kwargs)


@register_dataset(attrs=dict(unlisted=True))
def offline_sm(*args, **kwargs):
    kwargs.pop("data_ratio", None)
    return offline(*args, data_ratio=DatasetSizeRatio.SM, **kwargs)


@register_dataset(attrs=dict(unlisted=True))
def offline_md(*args, **kwargs):
    kwargs.pop("data_ratio", None)
    return offline(*args, data_ratio=DatasetSizeRatio.MD, **kwargs)
