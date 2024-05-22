import logging
import string
import numpy as np
from datasets import load_dataset

from ..registry import register_dataset, DatasetTag
from ..llm_data_utils import LMText, PromptFormat


__TASK_CATEGORIES = {
    "abstract_algebra": "STEM",
    "anatomy": "STEM",
    "astronomy": "STEM",
    "business_ethics": "Other",
    "clinical_knowledge": "Other",
    "college_biology": "STEM",
    "college_chemistry": "STEM",
    "college_computer_science": "STEM",
    "college_mathematics": "STEM",
    "college_medicine": "Other",
    "college_physics": "STEM",
    "computer_security": "STEM",
    "conceptual_physics": "STEM",
    "econometrics": "Social Sciences",
    "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM",
    "formal_logic": "Humanities",
    "global_facts": "Other",
    "high_school_biology": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM",
    "high_school_european_history": "Humanities",
    "high_school_geography": "Social Sciences",
    "high_school_government_and_politics": "Social Sciences",
    "high_school_macroeconomics": "Social Sciences",
    "high_school_mathematics": "STEM",
    "high_school_microeconomics": "Social Sciences",
    "high_school_physics": "STEM",
    "high_school_psychology": "Social Sciences",
    "high_school_statistics": "STEM",
    "high_school_us_history": "Humanities",
    "high_school_world_history": "Humanities",
    "human_aging": "Other",
    "human_sexuality": "Social Sciences",
    "international_law": "Humanities",
    "jurisprudence": "Humanities",
    "logical_fallacies": "Humanities",
    "machine_learning": "STEM",
    "management": "Other",
    "marketing": "Other",
    "medical_genetics": "Other",
    "miscellaneous": "Other",
    "moral_disputes": "Humanities",
    "moral_scenarios": "Humanities",
    "nutrition": "Other",
    "philosophy": "Humanities",
    "prehistory": "Humanities",
    "professional_accounting": "Other",
    "professional_law": "Humanities",
    "professional_medicine": "Other",
    "professional_psychology": "Social Sciences",
    "public_relations": "Social Sciences",
    "security_studies": "Social Sciences",
    "sociology": "Social Sciences",
    "us_foreign_policy": "Social Sciences",
    "virology": "Other",
    "world_religions": "Humanities",
}


def format_sample(sample, format):
    target_prompt = "\nAnswer:"

    question = sample["question"]
    answer_map = sample["choices"]
    target_idx = sample["answer"]

    if format == PromptFormat.CHOICE:
        context = "\n".join(
            [
                f"Question:\n{question}",
                "\nChoices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(
                        string.ascii_lowercase[: len(answer_map)], answer_map
                    )
                ],
            ]
        )

        target = string.ascii_lowercase[target_idx]
    elif format == PromptFormat.OE:
        context = f"Question: {question}"

        target = answer_map[target_idx]
    else:
        raise NotImplementedError(f"Unsupported prompt format {format}.")

    return LMText(context=context, target_prompt=target_prompt, target=target)


def format_sample_prompt(prompt_dataset, prompt_label, format, kshot=1, seed=None):
    if not kshot:
        return ""

    samples_idx = (
        np.random.default_rng(seed=seed)
        .permutation(len(prompt_dataset))[:kshot]
        .tolist()
    )

    fewshot_samples_prompt = [
        str(LMText.from_(prompt_dataset[idx])) + "\n" for idx in samples_idx
    ]

    if format == PromptFormat.CHOICE:
        prompt = [
            f"The following are multiple-choice questions (with answers) about {prompt_label}.\n",
            *fewshot_samples_prompt,
            "Now, answer the next question.\n\n",
        ]
    elif format == PromptFormat.OE:
        prompt = [
            f"The following are questions (with answers) about {prompt_label}.\n",
            *fewshot_samples_prompt,
            "Now, answer the next question.\n\n",
        ]
    else:
        raise NotImplementedError(f"Unsupported prompt format {format}.")

    return "\n".join(prompt)


def get_mmlu(
    task=None,
    prompt_style=None,
    eval_kshot=5,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    format = PromptFormat(prompt_style)

    dataset = load_dataset("cais/mmlu", task, trust_remote_code=True)
    if not use_cache:
        dataset.cleanup_cache_files()

    dataset = dataset.map(
        lambda sample: format_sample(sample, format).to_pydict(),
        num_proc=num_workers,
        remove_columns=dataset.column_names["test"],
    )

    prompt_label = (" ".join(task.split("_"))).capitalize()
    prompt_data = dataset.pop("dev")
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


@register_dataset(
    attrs=dict(task_categories=__TASK_CATEGORIES, tags=[DatasetTag.EVAL_ONLY])
)
def mmlu(*args, dataset_str=None, **kwargs):
    try:
        _, task = dataset_str.split(":")

        assert task in __TASK_CATEGORIES.keys()
    except ValueError:
        logging.exception(
            f'Dataset string should be formatted as "mmlu:<task>" (Got {dataset_str})',
        )
        raise
    except AssertionError:
        logging.exception(
            f'Task not found. Dataset string should be formatted as "mmlu:<task>" (Got {dataset_str})',
        )
        raise

    return get_mmlu(*args, **kwargs, task=task)


@register_dataset(attrs=dict(unlisted=True, collection=True))
def mmlu_all(*args, **kwargs):
    return [f"mmlu:{task}" for task in __TASK_CATEGORIES.keys()]
