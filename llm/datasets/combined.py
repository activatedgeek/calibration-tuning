import numpy as np
import logging
from datasets import concatenate_datasets

from .registry import get_dataset, list_datasets, register_dataset


def get_combined_train_dataset(
    max_n=100,
    root=None,
    tokenizer=None,
    seed=None,
    num_workers=8,
    use_dataset_cache=True,
    **_,
):
    all_datasets = list(
        filter(
            lambda x: ("combined" not in x) and ("mmlu" not in x) and ("bbh" not in x),
            list_datasets(),
        )
    )

    all_train_data, all_n = [], []
    a_val_data, a_test_data = None, None
    for dataset in all_datasets:
        train_data, val_data, test_data = get_dataset(
            dataset,
            root=root,
            tokenizer=tokenizer,
            seed=seed,
            num_workers=num_workers,
            use_cache=use_dataset_cache,
        )

        if train_data is not None:
            all_train_data.append(train_data)
            all_n.append(len(train_data))

        ## Use last val_data from any dataset as val.
        if val_data is not None and test_data is not None:
            logging.info(f"Setting validation/test data from '{dataset}'")
            a_val_data = val_data
            a_test_data = test_data

    max_n = min(max_n, sum(all_n))
    all_n = ((np.array(all_n) / max_n) * max_n).astype(int)

    all_train_data = concatenate_datasets(
        [
            train_data.shuffle(seed=seed).select(range(n))
            for train_data, n in zip(all_train_data, all_n)
        ]
    )

    return all_train_data, a_val_data, a_test_data


@register_dataset
def combined_100k(*args, **kwargs):
    return get_combined_train_dataset(*args, **kwargs, max_n=100_000)
