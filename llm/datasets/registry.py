from enum import Enum
import os
import logging
from functools import wraps
from pathlib import Path


__func_map = dict()
__attr_map = dict()


class DatasetTag(str, Enum):
    TRAIN_ONLY = "train_only"
    EVAL_ONLY = "eval_only"


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
        __attr_map[_wrapper.__name__] = attrs or dict()
        return _wrapper

    if function:
        return _decorator(function)
    return _decorator


dataset_key = lambda d: d.split(":")[0]


def get_dataset_attrs(name):
    key = dataset_key(name)
    if key not in __attr_map:
        raise ValueError(f'Dataset "{key}" not found.')

    return __attr_map[key]


def get_dataset_fn(name):
    key = dataset_key(name)
    if key not in __func_map:
        raise ValueError(f'Dataset "{key}" not found.')

    return __func_map[key]


def get_data_dir(data_dir=None):
    if data_dir is None:
        data_dir = (
            Path(os.environ.get("PROJECT_HOME", Path.home()))
            / Path.cwd().name
            / "datasets"
        )
    else:
        data_dir = Path(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)

    return str(data_dir.resolve())


def get_dataset(dataset, root=None, seed=42, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    dataset_fn = get_dataset_fn(dataset)

    root = get_data_dir(data_dir=root)

    if get_dataset_attrs(dataset).get("collection", False):
        return dataset_fn(root=root, dataset_str=dataset)

    train_data, val_data, test_data = dataset_fn(
        root=root,
        seed=seed,
        dataset_str=dataset,
        **kwargs,
    )

    info_str = " / ".join(
        [
            f"{s} (N = {len(ds)})"
            for ds, s in zip(
                (train_data, val_data, test_data), ("train", "validation", "test")
            )
            if ds is not None
        ]
    )
    logging.info(f'Loaded "{dataset}"; {info_str}')

    return train_data, val_data, test_data


def list_datasets():
    return [
        dname
        for dname in __func_map.keys()
        if not get_dataset_attrs(dname).get("unlisted", False)
    ]
