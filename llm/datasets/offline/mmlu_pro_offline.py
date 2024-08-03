import logging

from ..registry import register_dataset, DatasetTag
from ..hf.mmlu_pro import __TASKS
from .offline import get_offline


@register_dataset(attrs=dict(tasks=__TASKS, tags=[DatasetTag.EVAL_ONLY]))
def mmlu_pro_offline(
    root=None, dataset_str=None, prompt_style=None, eval_kshot=5, **kwargs
):
    try:
        _, name, task = dataset_str.split(":")

        assert task in __TASKS
    except ValueError:
        logging.exception(
            f'Dataset string should be formatted as "mmlu_pro_offline:<name>:<task>" (Got {dataset_str})',
        )
        raise
    except AssertionError:
        logging.exception(
            f'Task not found. Dataset string should be formatted as "mmlu_pro_offline:<name>:<task>" (Got {dataset_str})',
        )
        raise

    root = f"{root}/mmlu_pro_offline/{prompt_style}/{name}/{task}"

    return get_offline(root=root, eval_kshot=eval_kshot, **kwargs)


@register_dataset(attrs=dict(unlisted=True, collection=True))
def mmlu_pro_offline_all(dataset_str=None, **_):
    try:
        _, name = dataset_str.split(":")
    except ValueError:
        logging.exception(
            f'Dataset string should be formatted as "mmlu_pro_offline_all:<name>" (Got {dataset_str})',
        )
        raise

    return [f"mmlu_pro_offline:{name}:{task}" for task in __TASKS]
