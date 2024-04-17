import os
import logging
from functools import wraps
from pathlib import Path


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
    dataset_fn = get_dataset_fn(dataset.split(":")[0])

    root = get_data_dir(data_dir=root)

    train_data, val_data, test_data = dataset_fn(
        root=root,
        seed=seed,
        **{**kwargs, "dataset_str": dataset},
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
