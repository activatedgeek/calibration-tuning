import os
import string
import torch

from .registry import register_dataset
from .llm_utils import LMText


__all__ = [
    "get_arc",
]


def __format_sample(sample, tokenizer, style):
    target_prompt = "\nAnswer: "

    if style == "choice":
        question = sample["question"]
        answer_map = sample["choices"]["text"]

        context = "\n".join(
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
            ]
        )

        target = sample["answerKey"].lower() + tokenizer.eos_token
    elif style == "oe":
        question = sample["question"]
        answer_map = sample["choices"]["text"]

        context = "\n".join(
            [
                f"Question: {question}",
            ]
        )

        target = (
            answer_map[string.ascii_lowercase.index(sample["answerKey"].lower())]
            + tokenizer.eos_token
        )
    else:
        raise NotImplementedError

    return LMText(context=context, target_prompt=target_prompt, target=target)


def __generate_fewshot_prompts(
    tokenizer, prompt_style, prompt_dataset, kshot, seed=None
):
    if kshot <= 0:
        return ""

    fewshot_prompt = "\n".join(
        [
            "The following are questions with multiple choice answers.\n",
            *[
                str(__format_sample(prompt_dataset[idx], tokenizer, prompt_style))
                + "\n"
                for idx in torch.randperm(
                    len(prompt_dataset), generator=torch.Generator().manual_seed(seed)
                )[:kshot].tolist()
            ],
        ]
    )
    fewshot_prompt = fewshot_prompt + "\nNow, answer the next question."

    return fewshot_prompt


def __format_sample_with_prompt(
    sample, tokenizer, prompt_style, prompt_dataset, kshot, seed=None
):
    prompt = __generate_fewshot_prompts(
        tokenizer, prompt_style, prompt_dataset, kshot, seed=seed
    )
    if len(prompt):
        prompt += "\n\n"

    sample = __format_sample(sample, tokenizer, prompt_style)
    sample.prompt = prompt

    return sample


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
        data.filter(lambda x: x["answerKey"].lower() in string.ascii_lowercase).map(
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, k, seed=seed
            ).to_pydict(),
            num_proc=num_workers,
            remove_columns=[
                "id",
                "question",
                "choices",
                "answerKey",
            ],
        )
        for data, k in zip(
            [dataset.pop("train"), dataset.pop("validation"), dataset.pop("test")],
            [0, eval_kshot, eval_kshot],
        )
    ]

    return train_data, val_data, test_data


@register_dataset(attrs=dict(task_tags=["qa"]))
def arc(*args, prompt_style="choice", **kwargs):
    return get_arc(
        *args,
        **kwargs,
        subset="ARC-Easy",
        prompt_style=prompt_style,
    )


@register_dataset(attrs=dict(task_tags=["qa"]))
def arc_challenge(*args, prompt_style="choice", **kwargs):
    return get_arc(
        *args,
        **kwargs,
        subset="ARC-Challenge",
        prompt_style=prompt_style,
    )
