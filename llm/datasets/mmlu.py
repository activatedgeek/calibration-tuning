import os
import string
import torch

from .registry import register_dataset
from .llm_utils import LMText


__all__ = [
    "get_mmlu",
]


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

__ATTRS = dict(tasks=__TASKS)


def __format_sample(sample, tokenizer, style):
    target_prompt = "\nAnswer: "

    if style == "choice":
        question = sample["question"]
        choices = sample["choices"]

        context = "\n".join(
            [
                f"Question:\n{question}",
                "\nChoices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(string.ascii_lowercase[: len(choices)], choices)
                ],
            ]
        )

        target = string.ascii_lowercase[sample["answer"]] + tokenizer.eos_token
    elif style == "oe":
        question = sample["question"]

        context = f"Question:\n{question}"

        target = sample["choices"][sample["answer"]] + tokenizer.eos_token
    else:
        raise NotImplementedError

    return LMText(context=context, target_prompt=target_prompt, target=target)


def __generate_fewshot_prompts(
    tokenizer, prompt_style, prompt_dataset, instance, kshot, seed=None
):
    if kshot <= 0:
        return ""

    if prompt_style == "oe":
        fewshot_prompt = "\n".join(
            [
                f"The following are questions (with answers) about {' '.join(instance.split('_'))}.\n",
                *[
                    str(__format_sample(prompt_dataset[idx], tokenizer, prompt_style))
                    + "\n"
                    for idx in torch.randperm(
                        len(prompt_dataset),
                        generator=torch.Generator().manual_seed(seed),
                    )[:kshot].tolist()
                ],
            ]
        )
    elif prompt_style == "choice":
        fewshot_prompt = "\n".join(
            [
                f"The following are multiple choice questions (with answers) about {' '.join(instance.split('_'))}.\n",
                *[
                    str(__format_sample(prompt_dataset[idx], tokenizer, prompt_style))
                    + "\n"
                    for idx in torch.randperm(
                        len(prompt_dataset),
                        generator=torch.Generator().manual_seed(seed),
                    )[:kshot].tolist()
                ],
            ]
        )
    else:
        raise ValueError(f"Invalid prompt style: {prompt_style}")
    fewshot_prompt = fewshot_prompt + "\nNow, answer the next question."

    return fewshot_prompt


def __format_sample_with_prompt(
    sample, tokenizer, prompt_style, prompt_dataset, instance, kshot, seed=None
):
    prompt = __generate_fewshot_prompts(
        tokenizer, prompt_style, prompt_dataset, instance, kshot, seed=seed
    )
    if len(prompt):
        prompt += "\n\n"

    sample = __format_sample(sample, tokenizer, prompt_style)
    sample.prompt = prompt

    return sample


def get_mmlu(
    root=None,
    instance=None,
    prompt_style=None,
    eval_kshot=5,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "cais/mmlu", instance, cache_dir=os.environ.get("HF_DATASETS_CACHE", root)
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    dev_data = dataset.pop("dev")

    train_data, val_data, test_data = [
        dataset.pop(split).map(
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, dev_data, instance, k, seed=seed
            ).to_pydict(),
            num_proc=num_workers,
            remove_columns=[
                "question",
                "subject",
                "choices",
                "answer",
            ],
        )
        for split, k in zip(
            ["auxiliary_train", "validation", "test"], [0, eval_kshot, eval_kshot]
        )
    ]

    return train_data, val_data, test_data


@register_dataset(attrs=__ATTRS)
def mmlu(*args, dataset_str=None, prompt_style="choice", **kwargs):
    d, instance = dataset_str.split(":")

    assert d == "mmlu" and isinstance(
        instance, str
    ), f"Dataset string should be formatted as 'mmlu:<subset>', found {dataset_str}"

    assert instance in __TASKS, f"Task '{instance}' not found."

    return get_mmlu(
        *args,
        **kwargs,
        instance=instance,
        prompt_style=prompt_style,
    )
