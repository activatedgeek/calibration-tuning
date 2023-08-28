import os
import string
import torch

from .registry import register_dataset
from .llm_utils import tokenize_for_causal_lm


__all__ = [
    "get_arc",
]


def __format_prompt(sample, style, with_answer=False):
    if style == "choice":
        question = sample["question"]
        answer_map = sample["choices"]["text"]
        answer = sample["answerKey"].lower() + "</s>\n"

        prompt = "\n".join(
            [
                "Question:",
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

        return prompt

    raise NotImplementedError


def __generate_fewshot_prompts(dataset, prompt_style, kshot, seed=None):
    if kshot <= 0:
        return ""

    fewshot_prompt = "\n".join(
        [
            "The following are questions with multiple choice answers.\n",
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


def get_arc(
    root=None,
    subset=None,
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
        "ai2_arc", subset, cache_dir=os.environ.get("HF_DATASETS_CACHE", root)
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    train_data, val_data, test_data = [
        data.map(
            lambda x: {
                "source": __generate_fewshot_prompts(data, prompt_style, k, seed=seed)
                + __format_prompt(x, prompt_style),
                "target": f"{x['answerKey'].lower()}{tokenizer.eos_token}",
            },
            num_proc=num_workers,
            remove_columns=[
                "id",
                "question",
                "choices",
                "answerKey",
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


@register_dataset(attrs=dict(task_tags=["qa"]))
def arc(*args, **kwargs):
    return get_arc(
        *args,
        **kwargs,
        subset="ARC-Easy",
        prompt_style="choice",
    )


@register_dataset(attrs=dict(task_tags=["qa"]))
def arc_challenge(*args, **kwargs):
    return get_arc(
        *args,
        **kwargs,
        subset="ARC-Challenge",
        prompt_style="choice",
    )
