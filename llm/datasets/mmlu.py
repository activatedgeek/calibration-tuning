import os
import string
import torch

from .registry import register_dataset
from .llm_utils import tokenize_for_causal_lm


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


def __format_prompt(sample, style, with_answer=False):
    if style == "choice":
        question = sample["question"]
        choices = sample["choices"]
        answer = string.ascii_lowercase[sample["answer"]] + "</s>\n"

        prompt = "\n".join(
            [
                f"Question:\n{question}",
                "\nChoices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(string.ascii_lowercase[: len(choices)], choices)
                ],
                f"Answer: {answer if with_answer else ''}",
            ]
        )

        return prompt

    raise NotImplementedError


def __generate_fewshot_prompts(dataset, instance, prompt_style, kshot, seed=None):
    if kshot <= 0:
        return ""

    fewshot_prompt = "\n".join(
        [
            f"The following are multiple choice questions (with answers) about {' '.join(instance.split('_'))}.\n",
            *[
                __format_prompt(dataset[idx], prompt_style, with_answer=True)
                for idx in torch.randperm(
                    len(dataset), generator=torch.Generator().manual_seed(seed)
                )[:kshot].tolist()
            ],
        ]
    )
    fewshot_prompt = fewshot_prompt + "\nNow, answer the next question.\n\n"

    return fewshot_prompt


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
        dataset.pop(split)
        .map(
            lambda x: {
                "source": __generate_fewshot_prompts(
                    dev_data, instance, prompt_style, k, seed=seed
                )
                + __format_prompt(x, prompt_style),
                "target": f"{string.ascii_lowercase[x['answer']]}{tokenizer.eos_token}",
            },
            num_proc=num_workers,
            remove_columns=[
                "question",
                "choices",
                "answer",
            ],
        )
        .map(
            lambda x: tokenize_for_causal_lm(tokenizer, x),
            num_proc=num_workers,
            remove_columns=["source", "target"],
        )
        for split, k in zip(
            ["auxiliary_train", "validation", "test"], [0, eval_kshot, eval_kshot]
        )
    ]

    return train_data, val_data, test_data


@register_dataset(attrs=__ATTRS)
def mmlu(*args, dataset_str=None, **kwargs):
    d, instance = dataset_str.split(":")

    assert d == "mmlu" and isinstance(
        instance, str
    ), f"Dataset string should be formatted as 'mmlu:<subset>', found {dataset_str}"

    return get_mmlu(
        *args,
        **kwargs,
        instance=instance,
        prompt_style="choice",
    )
