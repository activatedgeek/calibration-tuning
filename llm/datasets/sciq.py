import os
import string
import torch
import numpy as np

from .registry import register_dataset
from .llm_utils import tokenize_for_causal_lm


__all__ = [
    "get_sciq",
]


def __format_prompt(sample, style, with_answer=False):
    if style == "choice":
        support = sample["support"]
        question = sample["question"]

        answer_order = np.random.permutation(4).tolist()
        answer_map = [
            sample["distractor1"],
            sample["distractor2"],
            sample["distractor3"],
            sample["correct_answer"],
        ]
        answer_map = [answer_map[idx] for idx in answer_order]

        answer = string.ascii_lowercase[answer_order.index(3)] + "</s>\n"

        prompt = "\n".join(
            [
                f"\n{support}\nQuestion:",
                question,
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

        if with_answer:
            return prompt

        return {"source": prompt, "target": answer}

    raise NotImplementedError


def __generate_fewshot_prompts(dataset, prompt_style, kshot=5, seed=None):
    if kshot <= 0:
        return ""

    fewshot_prompt = "\n".join(
        [
            "The following are multiple choice questions.\n",
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


def get_sciq(
    root=None,
    prompt_style=None,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset("sciq", cache_dir=os.environ.get("HF_DATASETS_CACHE", root))

    def __map(x, data, k):
        x_dict = __format_prompt(x, prompt_style)
        x_dict["source"] = (
            __generate_fewshot_prompts(data, prompt_style, kshot=k, seed=seed)
            + x_dict["source"]
        )
        x_dict["target"] = x_dict["target"].rstrip()
        return x_dict

    train_data, val_data, test_data = [
        data.map(
            lambda x: __map(x, data, k),
            num_proc=num_workers,
            remove_columns=[
                "question",
                "distractor1",
                "distractor2",
                "distractor3",
                "correct_answer",
                "support",
            ],
        ).map(
            lambda x: tokenize_for_causal_lm(tokenizer, x),
            num_proc=num_workers,
            remove_columns=["source", "target"],
        )
        for data, k in zip(
            [dataset.pop("train"), dataset.pop("validation"), dataset.pop("test")],
            [0, eval_kshot, eval_kshot],
        )
    ]

    return train_data, val_data, test_data


@register_dataset
def sciq(*args, **kwargs):
    return get_sciq(
        *args,
        **kwargs,
        prompt_style="choice",
    )
