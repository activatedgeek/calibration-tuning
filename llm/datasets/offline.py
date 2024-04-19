import logging
import os
import glob
from enum import Enum
import numpy as np
from datasets import load_dataset, Features, Value, DatasetDict

from .registry import register_dataset, get_dataset_attrs
from .llm_data_utils import LMText, LabeledStringDataCollator


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
    tokenizer=None,
    max_token_length=None,
    train_ratio=None,
    train_kshot=0,
    eval_kshot=0,
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

    if len(set(dataset.keys()) - set(embeddings.keys())) == 0:
        dataset = DatasetDict(
            {
                split: dataset[split].map(
                    lambda _, idx: {"embedding": embeddings[split][idx]},
                    with_indices=True,
                    num_proc=num_workers,
                )
                for split in data_files.keys()
            }
        )
        dataset = dataset.with_format(
            "np", columns=["embedding"], output_all_columns=True
        )

    prompt_kshot = {
        "train": train_kshot,
        "validation": eval_kshot,
        "test": eval_kshot,
    }

    data_splits = {
        split: (
            ds.remove_columns([c for c in ["prompt"] if c in ds.column_names])
            if prompt_kshot[split] == 0
            else ds
        )
        for split, ds in dataset.items()
    }

    if max_token_length is not None:
        tokenizer_args = LabeledStringDataCollator.get_tokenizer_args(tokenizer)

        def token_length_filter(instance):
            inputs = tokenizer(
                [str(LMText.from_(instance))],
                **tokenizer_args,
            )
            return inputs.get("input_ids").size(-1) <= max_token_length

        data_splits = {
            k: ds.filter(token_length_filter, num_proc=num_workers)
            for k, ds in data_splits.items()
        }

    train_data = data_splits.pop("train", None)
    if train_data is not None and train_ratio is not None:
        n_train = len(train_data)
        idxs = np.random.default_rng(seed=seed).choice(
            range(n_train), int(DatasetSizeRatio(train_ratio) * n_train)
        )
        train_data = train_data.select(idxs)

    val_data = data_splits.pop("validation", None)
    test_data = data_splits.pop("test", None)

    return train_data, val_data, test_data


@register_dataset
def offline(*args, root=None, dataset_str=None, prompt_style=None, **kwargs):
    try:
        _, name = dataset_str.split(":")
    except ValueError:
        logging.exception(
            f'Dataset string should be formatted as "offline:<name>" (Got {dataset_str})',
            exc_info=True,
        )
        raise

    root = f"{root}/offline/{name}-{prompt_style}"

    return get_offline(*args, root=root, **kwargs)


@register_dataset
def offline_xxs(*args, **kwargs):
    kwargs.pop("train_ratio", None)
    return offline(*args, train_ratio=DatasetSizeRatio.XXS, **kwargs)


@register_dataset
def offline_xs(*args, **kwargs):
    kwargs.pop("train_ratio", None)
    return offline(*args, train_ratio=DatasetSizeRatio.XS, **kwargs)


@register_dataset
def offline_sm(*args, **kwargs):
    kwargs.pop("train_ratio", None)
    return offline(*args, train_ratio=DatasetSizeRatio.SM, **kwargs)


@register_dataset
def offline_md(*args, **kwargs):
    kwargs.pop("train_ratio", None)
    return offline(*args, train_ratio=DatasetSizeRatio.MD, **kwargs)


@register_dataset(attrs=get_dataset_attrs("mmlu"))
def mmlu_offline(
    *args, root=None, dataset_str=None, prompt_style=None, eval_kshot=5, **kwargs
):
    try:
        _, name, task = dataset_str.split(":")

        assert task in get_dataset_attrs("mmlu").get("tasks")
    except ValueError:
        logging.exception(
            f'Dataset string should be formatted as "mmlu_offline:<name>:<task>" (Got {dataset_str})',
            exc_info=True,
        )
        raise
    except AssertionError:
        logging.exception(
            f'Task not found. Dataset string should be formatted as "mmlu_offline:<name>:<task>" (Got {dataset_str})',
            exc_info=True,
        )
        raise

    root = f"{root}/mmlu_offline/{prompt_style}/{name}/{task}"

    return get_offline(*args, root=root, eval_kshot=eval_kshot, **kwargs)
