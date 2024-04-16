import string
import numpy as np
from datasets import load_dataset

from ..registry import register_dataset
from ..llm_utils import LMText, PromptFormat


__TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


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


@register_dataset(attrs=dict(tasks=__TASKS, eval=True))
def mmlu(*args, dataset_str=None, **kwargs):
    _, task = dataset_str.split(":")

    assert (
        task in __TASKS
    ), f'Dataset string should be formatted as "mmlu:<task>" (Got {dataset_str}). "{task}" task not found.'

    return get_mmlu(*args, **kwargs, task=task)
