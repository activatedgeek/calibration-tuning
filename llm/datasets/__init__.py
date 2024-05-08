from .registry import (
    register_dataset,
    get_data_dir,
    get_dataset,
    get_dataset_attrs,
    list_datasets,
)
from .utils import get_loader, get_num_workers

from .llm_data_utils import (
    IGNORE_LABEL,
    get_token_vec,
    LMText,
    LabeledStringDataCollator,
)
from .llm_utils_oe import prepare_uncertainty_query


__all__ = [
    "register_dataset",
    "get_data_dir",
    "get_dataset",
    "get_dataset_attrs",
    "list_datasets",
    "get_loader",
    "get_num_workers",
    "IGNORE_LABEL",
    "LabeledStringDataCollator",
    "get_token_vec",
    "LMText",
    "prepare_uncertainty_query",
]


def __setup():
    from importlib import import_module

    for n in [
        "hf",
        "offline",
    ]:
        import_module(f".{n}", __name__)


__setup()
