import logging
from datasets import load_dataset

from ..registry import register_dataset, DatasetTag
from ..llm_data_utils import LMText, PromptFormat
from .mmlu import format_sample, format_sample_prompt


__TASKS = [
    "biology",
    "business",
    "chemistry",
    "computer_science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
]


def get_mmlu_pro(
    task=None,
    prompt_style=None,
    eval_kshot=5,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    format = PromptFormat(prompt_style)
    task = " ".join(task.split("_"))

    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    if not use_cache:
        dataset.cleanup_cache_files()

    dataset = (
        dataset.filter(lambda s: s["category"] == task)
        .rename_columns(
            {
                "options": "choices",
                "answer": "answer_choice",
                "answer_index": "answer",
            }
        )
        .remove_columns(
            ["question_id", "answer_choice", "cot_content", "category", "src"]
        )
    )

    dataset = dataset.map(
        lambda sample: format_sample(sample, format).to_pydict(),
        num_proc=num_workers,
        remove_columns=dataset.column_names["test"],
    )

    prompt_label = (" ".join(task.split("_"))).capitalize()
    prompt_data = dataset.pop("validation")
    prompt_kshot = {
        "validation": eval_kshot,
        "test": eval_kshot,
    }

    data_splits = {
        split: ds.map(
            lambda _, idx: {
                "prompt": format_sample_prompt(
                    prompt_data,
                    prompt_label,
                    format,
                    kshot=prompt_kshot[split],
                    seed=seed + idx,
                )
            },
            with_indices=True,
            num_proc=num_workers,
        )
        for split, ds in dataset.items()
    }

    train_data = data_splits.pop("train", None)
    val_data = data_splits.pop("validation", None)
    test_data = data_splits.pop("test", None)

    return train_data, val_data, test_data


@register_dataset(attrs=dict(task_categories=__TASKS, tags=[DatasetTag.EVAL_ONLY]))
def mmlu_pro(*args, dataset_str=None, **kwargs):
    try:
        _, task = dataset_str.split(":")

        assert task in __TASKS
    except ValueError:
        logging.exception(
            f'Dataset string should be formatted as "mmlu_pro:<task>" (Got {dataset_str})',
        )
        raise
    except AssertionError:
        logging.exception(
            f'Task not found. Dataset string should be formatted as "mmlu_pro:<task>" (Got {dataset_str})',
        )
        raise

    return get_mmlu_pro(*args, **kwargs, task=task)


@register_dataset(attrs=dict(unlisted=True, collection=True))
def mmlu_pro_all(*args, **kwargs):
    return [f"mmlu_pro:{task}" for task in __TASKS]
