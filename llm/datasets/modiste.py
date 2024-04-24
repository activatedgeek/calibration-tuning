import logging
import json
from pathlib import Path
from datasets import Dataset

from .registry import register_dataset


__ATTRS = dict(
    tasks=[
        "us_foreign_policy",
        "high_school_chemistry",
        "high_school_us_history",
        "marketing",
        "high_school_computer_science",
        "virology",
        "high_school_biology",
        "astronomy",
        "elementary_mathematics",
        "high_school_geography",
        "logical_fallacies",
        "management",
    ]
)


def get_modiste(root=None, task=None, **_):
    with open(Path(root) / "mmlu_llm_gens.json") as f:
        data = json.load(f)

    dataset = Dataset.from_list(data[task])

    dataset = dataset.remove_columns(column_names=["Idx", "Embedding"])

    column_map = {"Prompt": "context", "Answer": "target", "LLM Answer": "output"}
    dataset = dataset.rename_columns(column_mapping=column_map)

    return None, None, dataset


@register_dataset(attrs=__ATTRS)
def modiste(*args, root=None, dataset_str=None, **kwargs):
    root = Path(root) / "modiste-example-turing"

    try:
        _, task = dataset_str.split(":")

        assert task in __ATTRS["tasks"]
    except ValueError:
        logging.exception(
            f'Dataset string should be formatted as "modiste:<task>" (Got {dataset_str})',
            exc_info=True,
        )
        raise

    return get_modiste(*args, root=root, task=task, **kwargs)
