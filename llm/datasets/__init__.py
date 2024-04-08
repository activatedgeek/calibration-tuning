from .registry import register_dataset, get_dataset, get_dataset_attrs, list_datasets
from .utils import get_loader, get_num_workers
from .combined import get_all_datasets_list

from .llm_utils import IGNORE_LABEL, get_token_vec, LMText
from .llm_utils_oe import prepare_uncertainty_query
from .data_collator import LabeledStringDataCollator


__all__ = [
    "register_dataset",
    "get_dataset",
    "get_dataset_attrs",
    "list_datasets",
    "get_loader",
    "get_num_workers",
    "get_all_datasets_list",
    "IGNORE_LABEL",
    "LabeledStringDataCollator",
    "get_token_vec",
    "LMText",
    "prepare_uncertainty_query",
]


def __setup():
    from importlib import import_module

    for n in [
        "arc",
        "boolq",
        "combined",
        "commonsense_qa",
        "cosmos_qa",
        "gsm8k",
        "hellaswag",
        "math_qa",
        "mmlu",
        "mmlu_oe_offline",
        "nli",
        "obqa",
        "offline",
        "piqa",
        "sciq",
        "siqa",
        "story_cloze",
        "super_glue",
        "trec",
        "truthful_qa",
        "winogrande",
        "wsc",
    ]:
        import_module(f".{n}", __name__)


__setup()
