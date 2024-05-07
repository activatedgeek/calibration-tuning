import logging
import json
from pathlib import Path
from datasets import Dataset, Value

from ..registry import register_dataset, DatasetTag
from ..llm_data_utils import LMText


__TASKS = [
    "elementary_mathematics",
    "high_school_biology",
    "us_foreign_policy",
    "high_school_computer_science",
]


def format_sample(sample):
    context = sample["prompt"]
    target = sample["label"]
    output = sample["llm_answer"]

    return LMText(
        context=context,
        target=target,
        output=output,
    )


def get_modiste(root=None, task=None, num_workers=8, use_cache=True, **_):
    with open(Path(root) / "mmlu_responses_w_conf.json") as f:
        data = json.load(f)

    dataset = Dataset.from_list(data[task])
    if not use_cache:
        dataset.cleanup_cache_files()

    dataset = dataset.map(
        lambda sample: {
            **format_sample(sample).to_pydict(),
            ## Keep IDs for mapping later.
            "example_idx": sample["example_idx"],
            "orig_example_idx": sample["orig_example_idx"],
        },
        num_proc=num_workers,
        remove_columns=list(
            set(dataset.column_names) - set(["example_idx", "orig_example_idx"])
        ),
    )

    types = dataset.features.copy()
    types["example_idx"] = Value("int64")
    types["orig_example_idx"] = Value("int64")
    dataset = dataset.cast(types, num_proc=num_workers)

    return None, None, dataset


@register_dataset(attrs=dict(tasks=__TASKS, tags=[DatasetTag.EVAL_ONLY]))
def modiste_mmlu(*args, root=None, dataset_str=None, **kwargs):
    root = Path(root) / "modiste"

    try:
        _, task = dataset_str.split(":")

        assert task in __TASKS
    except ValueError:
        logging.exception(
            f'Dataset string should be formatted as "modiste_mmlu:<task>" (Got {dataset_str})',
        )
        raise
    except AssertionError:
        logging.exception(
            f'Task not found. Dataset string should be formatted as "modiste_mmlu:<task>" (Got {dataset_str})',
        )
        raise

    return get_modiste(*args, root=root, task=task, **kwargs)


@register_dataset(attrs=dict(unlisted=True, collection=True))
def modiste_mmlu_all(*args, **kwargs):
    return [f"modiste_mmlu:{task}" for task in __TASKS]
