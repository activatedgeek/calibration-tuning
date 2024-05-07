import logging

from ..registry import register_dataset, get_dataset_attrs, DatasetTag
from ..hf.mmlu import __TASKS
from .offline import get_offline


@register_dataset(attrs=dict(tasks=__TASKS, tags=[DatasetTag.EVAL_ONLY]))
def mmlu_offline(
    *args, root=None, dataset_str=None, prompt_style=None, eval_kshot=5, **kwargs
):
    try:
        _, name, task = dataset_str.split(":")

        assert task in get_dataset_attrs("mmlu").get("tasks")
    except ValueError:
        logging.exception(
            f'Dataset string should be formatted as "mmlu_offline:<name>:<task>" (Got {dataset_str})',
        )
        raise
    except AssertionError:
        logging.exception(
            f'Task not found. Dataset string should be formatted as "mmlu_offline:<name>:<task>" (Got {dataset_str})',
        )
        raise

    root = f"{root}/mmlu_offline/{prompt_style}/{name}/{task}"

    return get_offline(*args, root=root, eval_kshot=eval_kshot, **kwargs)


@register_dataset(attrs=dict(unlisted=True, collection=True))
def mmlu_offline_all(*args, dataset_str=None, **kwargs):
    try:
        _, name = dataset_str.split(":")
    except ValueError:
        logging.exception(
            f'Dataset string should be formatted as "mmlu_offline_all:<name>" (Got {dataset_str})',
        )
        raise

    return [f"mmlu_offline:{name}:{task}" for task in __TASKS]
