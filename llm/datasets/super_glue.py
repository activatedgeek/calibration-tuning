import os
import string
import torch

from .registry import register_dataset
from .llm_utils import tokenize_for_causal_lm


__all__ = [
    "get_cb",
    "get_multirc",
    "get_copa",
]


def get_cb(
    root=None,
    prompt_style=None,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "super_glue", "cb", cache_dir=os.environ.get("HF_DATASETS_CACHE", root)
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    def __format_prompt(sample, style, with_answer=False):
        if style == "choice":
            premise = sample["premise"]
            hypothesis = sample["hypothesis"]
            answer_map = ["Yes", "No", "It's impossible to say"]
            answer = string.ascii_lowercase[sample["label"]] + "</s>\n"

            prompt = "\n".join(
                [
                    "Premise:",
                    premise,
                    "\nHypothesis:",
                    hypothesis,
                    "\nChoices:",
                    *[
                        f"  ({n}): {c}"
                        for n, c in zip(
                            string.ascii_lowercase[: len(answer_map)], answer_map
                        )
                    ],
                    f"Answer: {answer if with_answer else ''}",
                ]
            )

            return prompt

        raise NotImplementedError

    def __generate_fewshot_prompts(d, style, kshot, seed=None):
        if kshot <= 0:
            return ""

        fewshot_prompt = "\n".join(
            [
                "The following are multiple choice questions (with premise, hypothesis, and answers) about entailment.\n",
                *[
                    __format_prompt(d[idx], style, with_answer=True)
                    for idx in torch.randperm(
                        len(d), generator=torch.Generator().manual_seed(seed)
                    )[:kshot].tolist()
                ],
            ]
        )
        fewshot_prompt = fewshot_prompt + "\nNow, answer the following.\n\n"

        return fewshot_prompt

    data_splits = [
        data.filter(lambda x: x["label"] in [0, 1, 2], num_proc=num_workers)
        for data in [
            dataset.pop("train"),
            dataset.pop("validation"),
        ]
    ]
    train_data, val_data = [
        data.map(
            lambda x: {
                "source": __generate_fewshot_prompts(data, prompt_style, k, seed=seed)
                + __format_prompt(x, prompt_style),
                "target": f"{string.ascii_lowercase[x['label']]}{tokenizer.eos_token}",
            },
            num_proc=num_workers,
            remove_columns=[
                "premise",
                "hypothesis",
                "idx",
                "label",
            ],
        ).map(
            lambda x: tokenize_for_causal_lm(tokenizer, x),
            num_proc=num_workers,
            remove_columns=["source", "target"],
        )
        for data, k in zip(data_splits, [0, eval_kshot])
    ]

    return train_data, val_data, None


@register_dataset(attrs=dict(task_tags=["entailment"]))
def cb(*args, **kwargs):
    return get_cb(
        *args,
        **kwargs,
        prompt_style="choice",
    )


def get_multirc(
    root=None,
    prompt_style=None,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "super_glue", "multirc", cache_dir=os.environ.get("HF_DATASETS_CACHE", root)
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    def __format_prompt(sample, style, with_answer=False):
        if style == "choice":
            paragraph = sample["paragraph"]
            question = sample["question"]
            answer_map = ["No", "Yes"]
            answer = string.ascii_lowercase[sample["label"]] + "</s>\n"

            prompt = "\n".join(
                [
                    "Paragraph:",
                    paragraph,
                    f"\nQ: {question}",
                    f"\nA: {sample['answer']}" "\n\nChoices:",
                    *[
                        f"  ({n}): {c}"
                        for n, c in zip(
                            string.ascii_lowercase[: len(answer_map)], answer_map
                        )
                    ],
                    f"Answer: {answer if with_answer else ''}",
                ]
            )

            return prompt

        raise NotImplementedError

    def __generate_fewshot_prompts(d, style, kshot, seed=None):
        if kshot <= 0:
            return ""

        fewshot_prompt = "\n".join(
            [
                "The following are reading comprehensions (with answers).\n",
                *[
                    __format_prompt(d[idx], style, with_answer=True)
                    for idx in torch.randperm(
                        len(d), generator=torch.Generator().manual_seed(seed)
                    )[:kshot].tolist()
                ],
            ]
        )
        fewshot_prompt = fewshot_prompt + "\nNow, answer the following.\n\n"

        return fewshot_prompt

    data_splits = [
        data.filter(lambda x: x["label"] in [0, 1, 2], num_proc=num_workers)
        for data in [
            dataset.pop("train"),
            dataset.pop("validation"),
        ]
    ]
    train_data, val_data = [
        data.map(
            lambda x: {
                "source": __generate_fewshot_prompts(data, prompt_style, k, seed=seed)
                + __format_prompt(x, prompt_style),
                "target": f"{string.ascii_lowercase[x['label']]}{tokenizer.eos_token}",
            },
            num_proc=num_workers,
            remove_columns=[
                "paragraph",
                "question",
                "answer",
                "idx",
                "label",
            ],
        ).map(
            lambda x: tokenize_for_causal_lm(tokenizer, x),
            num_proc=num_workers,
            remove_columns=["source", "target"],
        )
        for data, k in zip(data_splits, [0, eval_kshot])
    ]

    return train_data, val_data, None


@register_dataset(attrs=dict(task_tags=["comprehension"]))
def multirc(*args, **kwargs):
    return get_multirc(
        *args,
        **kwargs,
        prompt_style="choice",
    )


def get_copa(
    root=None,
    prompt_style=None,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "super_glue", "copa", cache_dir=os.environ.get("HF_DATASETS_CACHE", root)
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    def __format_prompt(sample, style, with_answer=False):
        if style == "choice":
            premise = sample["premise"]
            answer_map = [sample["choice1"], sample["choice2"]]
            answer = string.ascii_lowercase[sample["label"]] + "</s>\n"

            prompt = "\n".join(
                [
                    "Premise:",
                    premise,
                    "\nChoices:",
                    *[
                        f"  ({n}): {c}"
                        for n, c in zip(
                            string.ascii_lowercase[: len(answer_map)], answer_map
                        )
                    ],
                    f"Answer: {answer if with_answer else ''}",
                ]
            )

            return prompt

        raise NotImplementedError

    def __generate_fewshot_prompts(d, style, kshot, seed=None):
        if kshot <= 0:
            return ""

        fewshot_prompt = "\n".join(
            [
                "The following are reading comprehensions (with answers).\n",
                *[
                    __format_prompt(d[idx], style, with_answer=True)
                    for idx in torch.randperm(
                        len(d), generator=torch.Generator().manual_seed(seed)
                    )[:kshot].tolist()
                ],
            ]
        )
        fewshot_prompt = fewshot_prompt + "\nNow, answer the following.\n\n"

        return fewshot_prompt

    data_splits = [
        data.filter(lambda x: x["label"] in [0, 1, 2], num_proc=num_workers)
        for data in [
            dataset.pop("train"),
            dataset.pop("validation"),
        ]
    ]
    train_data, val_data = [
        data.map(
            lambda x: {
                "source": __generate_fewshot_prompts(data, prompt_style, k, seed=seed)
                + __format_prompt(x, prompt_style),
                "target": f"{string.ascii_lowercase[x['label']]}{tokenizer.eos_token}",
            },
            num_proc=num_workers,
            remove_columns=[
                "premise",
                "choice1",
                "choice2",
                "question",
                "idx",
                "label",
            ],
        ).map(
            lambda x: tokenize_for_causal_lm(tokenizer, x),
            num_proc=num_workers,
            remove_columns=["source", "target"],
        )
        for data, k in zip(data_splits, [0, eval_kshot])
    ]

    return train_data, val_data, None


@register_dataset(attrs=dict(task_tags=["commonsense"]))
def copa(*args, **kwargs):
    return get_copa(
        *args,
        **kwargs,
        prompt_style="choice",
    )