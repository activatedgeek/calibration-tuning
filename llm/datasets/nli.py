import os
import string
import torch

from .registry import register_dataset
from .llm_utils import tokenize_for_causal_lm


__all__ = [
    "get_snli",
    "get_anli",
]


def __format_prompt(sample, style, with_answer=False):
    if style == "choice":
        premise = sample["premise"]
        hypothesis = sample["hypothesis"]
        answer_map = {0: "Yes", 1: "No", 2: "It's impossible to say"}
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
                        string.ascii_lowercase[: len(answer_map)], answer_map.values()
                    )
                ],
                f"Answer: {answer if with_answer else ''}",
            ]
        )

        return prompt

    raise NotImplementedError


def __generate_fewshot_prompts(dataset, prompt_style, kshot=5, seed=None):
    if kshot <= 0:
        return ""

    fewshot_prompt = "\n".join(
        [
            "The following are multiple choice questions (with premise, hypothesis, and answers) about entailment.\n",
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


def get_snli(
    root=None,
    prompt_style=None,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset("snli", cache_dir=os.environ.get("HF_DATASETS_CACHE", root))

    data_splits = [
        data.filter(lambda x: x["label"] in [0, 1, 2], num_proc=num_workers)
        for data in [
            dataset.pop("train"),
            dataset.pop("validation"),
            dataset.pop("test"),
        ]
    ]
    train_data, val_data, test_data = [
        data.map(
            lambda x: {
                "source": __generate_fewshot_prompts(
                    data, prompt_style, kshot=k, seed=seed
                )
                + __format_prompt(x, prompt_style),
                "target": f"{string.ascii_lowercase[x['label']]}{tokenizer.eos_token}",
            },
            num_proc=num_workers,
            remove_columns=[
                "premise",
                "hypothesis",
                "label",
            ],
        ).map(
            lambda x: tokenize_for_causal_lm(tokenizer, x),
            num_proc=num_workers,
            remove_columns=["source", "target"],
        )
        for data, k in zip(data_splits, [0, eval_kshot, eval_kshot])
    ]

    return train_data, val_data, test_data


def get_anli(
    root=None,
    round=None,
    prompt_style=None,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    **_,
):
    from datasets import load_dataset

    assert round in [1, 2, 3], "Invalid round value"

    dataset = load_dataset("anli", cache_dir=os.environ.get("HF_DATASETS_CACHE", root))

    dev_data = dataset.pop(f"dev_r{round}")

    data_splits = [
        data.filter(lambda x: x["label"] in [0, 1, 2], num_proc=num_workers)
        for data in [dataset.pop(f"train_r{round}"), dataset.pop(f"test_r{round}")]
    ]
    train_data, test_data = [
        data.map(
            lambda x: {
                "source": __generate_fewshot_prompts(
                    dev_data, prompt_style, kshot=k, seed=seed
                )
                + __format_prompt(x, prompt_style),
                "target": f"{string.ascii_lowercase[x['label']]}{tokenizer.eos_token}",
            },
            num_proc=num_workers,
            remove_columns=[
                "premise",
                "hypothesis",
                "label",
            ],
        ).map(
            lambda x: tokenize_for_causal_lm(tokenizer, x),
            num_proc=num_workers,
            remove_columns=["source", "target"],
        )
        for data, k in zip(data_splits, [0, eval_kshot])
    ]

    return train_data, None, test_data


@register_dataset
def snli(*args, **kwargs):
    return get_snli(
        *args,
        **kwargs,
        prompt_style="choice",
    )


@register_dataset
def anli_r1(*args, **kwargs):
    return get_anli(
        *args,
        **kwargs,
        round=1,
        prompt_style="choice",
    )


@register_dataset
def anli_r2(*args, **kwargs):
    return get_anli(
        *args,
        **kwargs,
        round=2,
        prompt_style="choice",
    )


@register_dataset
def anli_r3(*args, **kwargs):
    return get_anli(
        *args,
        **kwargs,
        round=3,
        prompt_style="choice",
    )
