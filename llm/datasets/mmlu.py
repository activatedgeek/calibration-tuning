import os
import string

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

__ATTRS = dict(label2char=lambda idx: string.ascii_lowercase[idx], tasks=__TASKS)


def __format_prompt(sample, style, with_answer=False):
    if style == "mcq":
        question = sample["question"]
        choices = sample["choices"]
        answer = string.ascii_lowercase[sample["answer"]] + "\n"

        prompt = "\n".join(
            [
                f"Question: {question}",
                "Choices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(string.ascii_lowercase[: len(choices)], choices)
                ],
                f"Answer: {answer if with_answer else ''}",
            ]
        )

        return prompt

    raise NotImplementedError


def get_mmlu(
    root=None,
    instance=None,
    prompt_style=None,
    kshot=5,
    seed=None,
    tokenizer=None,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "cais/mmlu", instance, cache_dir=os.environ.get("DATADIR", root)
    )

    if kshot > 0:
        fewshot_prompt = "\n".join(
            [
                f"The following are multiple choice questions (with answers) about {' '.join(instance.split('_'))}.\n",
                *[
                    __format_prompt(dataset["dev"][idx], prompt_style, with_answer=True)
                    for idx in range(kshot)
                ],
            ]
        )
        fewshot_prompt = fewshot_prompt + "\nNow, answer the next "
    else:
        fewshot_prompt = ""

    dataset = dataset.map(
        lambda x: {
            "source": fewshot_prompt + __format_prompt(x, prompt_style),
            "target": f"{string.ascii_lowercase[x['answer']]}{tokenizer.eos_token}",
        },
        num_proc=4,
        remove_columns=[
            "question",
            "choices",
            "answer",
        ],
    ).map(
        lambda x: tokenize_for_causal_lm(tokenizer, x),
        num_proc=4,
        remove_columns=["source", "target"],
    )

    train_data = dataset["auxiliary_train"].shuffle(seed=seed)
    val_data = dataset["validation"].shuffle(seed=seed)
    test_data = dataset["test"].shuffle(seed=seed)

    return train_data, val_data, test_data


@register_dataset(attrs=__ATTRS)
def mmlu(*args, instance=None, **kwargs):
    return get_mmlu(
        *args,
        **kwargs,
        instance=instance,
        prompt_style="mcq",
    )
