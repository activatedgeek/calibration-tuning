import logging
from functools import wraps
import numpy as np
import torch
from torch.utils.data import Subset

from .utils import get_data_dir, LabelNoiseDataset

__all__ = [
    "register_dataset",
    "get_dataset",
    "get_dataset_attrs",
]


__func_map = dict()
__attr_map = dict()


def register_dataset(function=None, attrs=None, **d_kwargs):
    def _decorator(f):
        @wraps(f)
        def _wrapper(*args, **kwargs):
            all_kwargs = {**d_kwargs, **kwargs}
            return f(*args, **all_kwargs)

        assert (
            _wrapper.__name__ not in __func_map
        ), f'Duplicate registration for "{_wrapper.__name__}"'

        __func_map[_wrapper.__name__] = _wrapper
        __attr_map[_wrapper.__name__] = attrs
        return _wrapper

    if function:
        return _decorator(function)
    return _decorator


def get_dataset_fn(name):
    if name not in __func_map:
        raise ValueError(f'Dataset "{name}" not found.')

    return __func_map[name]


def get_dataset_attrs(name):
    if name not in __attr_map:
        raise ValueError(f'Dataset "{name}" attributes not found.')

    return __attr_map[name]


def list_datasets():
    return list(__func_map.keys())


def get_dataset(dataset, root=None, seed=42, train_subset=1, label_noise=0, **kwargs):
    dataset_fn = get_dataset_fn(dataset.split(":")[0])

    root = get_data_dir(data_dir=root)

    train_data, val_data, test_data = dataset_fn(
        root=root, seed=seed, dataset_str=dataset, **kwargs
    )

    if label_noise > 0:
        train_data = LabelNoiseDataset(
            train_data,
            n_labels=get_dataset_attrs(dataset).get("num_classes"),
            label_noise=label_noise,
            seed=seed,
        )

    if np.abs(train_subset) < 1:
        n = len(train_data)
        ns = int(n * np.abs(train_subset))

        ## NOTE: -ve train_subset fraction to get latter segment.
        randperm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        randperm = randperm[ns:] if train_subset < 0 else randperm[:ns]

        train_data = Subset(train_data, randperm)

    logging.info(f'Loaded "{dataset}".')

    return train_data, val_data, test_data
