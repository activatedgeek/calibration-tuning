import logging

from ..registry import register_dataset, DatasetTag
from ..hf.mmlu import __TASK_CATEGORIES
from .offline import get_offline
from .offline_logits import get_offline_logits


@register_dataset(
    attrs=dict(task_categories=__TASK_CATEGORIES, tags=[DatasetTag.EVAL_ONLY])
)
def mmlu_offline(
    root=None, dataset_str=None, prompt_style=None, eval_kshot=5, **kwargs
):
    try:
        _, name, task = dataset_str.split(":")

        assert task in __TASK_CATEGORIES.keys()
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

    return get_offline(root=root, eval_kshot=eval_kshot, **kwargs)


@register_dataset(attrs=dict(unlisted=True, collection=True))
def mmlu_offline_all(dataset_str=None, **_):
    try:
        _, name = dataset_str.split(":")
    except ValueError:
        logging.exception(
            f'Dataset string should be formatted as "mmlu_offline_all:<name>" (Got {dataset_str})',
        )
        raise

    return [f"mmlu_offline:{name}:{task}" for task in __TASK_CATEGORIES.keys()]


@register_dataset(
    attrs=dict(task_categories=__TASK_CATEGORIES, tags=[DatasetTag.EVAL_ONLY])
)
def mmlu_offline_query_logits(root=None, dataset_str=None, **kwargs):
    name, kind, dataset = dataset_str.split(":")

    root = f"{root}/{name}/{kind}/{dataset}"

    return get_offline_logits(root=root, **kwargs)


@register_dataset(attrs=dict(unlisted=True, collection=True))
def mmlu_offline_query_logits_all(dataset_str=None, **_):
    try:
        _, name = dataset_str.split(":")
    except ValueError:
        logging.exception(
            f'Dataset string should be formatted as "mmlu_offline_query_logits_all:<name>" (Got {dataset_str})',
        )
        raise

    return [
        f"mmlu_offline_query_logits:{name}:{task}" for task in __TASK_CATEGORIES.keys()
    ]


@register_dataset(
    attrs=dict(task_categories=__TASK_CATEGORIES, tags=[DatasetTag.EVAL_ONLY])
)
def mmlu_offline_ve_logits(root=None, dataset_str=None, **kwargs):
    name, kind, dataset = dataset_str.split(":")

    root = f"{root}/{name}/{kind}/{dataset}"

    return get_offline_logits(root=root, **kwargs)


@register_dataset(attrs=dict(unlisted=True, collection=True))
def mmlu_offline_ve_logits_all(dataset_str=None, **_):
    try:
        _, name = dataset_str.split(":")
    except ValueError:
        logging.exception(
            f'Dataset string should be formatted as "mmlu_offline_ve_logits_all:<name>" (Got {dataset_str})',
        )
        raise

    return [
        f"mmlu_offline_ve_logits:{name}:{task}" for task in __TASK_CATEGORIES.keys()
    ]
