import string
import torch
from datasets import load_dataset

from .registry import register_dataset
from .llm_utils import LMText


__all__ = [
    "get_arc",
]


def __format_sample(sample, tokenizer, style):
    target_prompt = "\nAnswer:"

    sample["answerKey"] = sample["answerKey"].lower()

    if sample["answerKey"] in string.ascii_lowercase:
        sample["answerKey"] = string.ascii_lowercase.index(sample["answerKey"])
    else:
        sample["answerKey"] = int(sample["answerKey"]) - 1

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

        target = string.ascii_lowercase[sample["answerKey"]]
    elif style == "oe":
        question = sample["question"]
        answer_map = sample["choices"]["text"]

        context = "\n".join(
            [
                f"The question is: {question}",
            ]
        )

        target = answer_map[sample["answerKey"]]
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
            "The following are questions with answers.\n",
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
    if kshot > 0:
        prompt = (
            __generate_fewshot_prompts(
                tokenizer, prompt_style, prompt_dataset, kshot, seed=seed
            )
            + "\n\n"
        )
    else:
        if prompt_style == "oe":
            prompt = (
                "\n".join(
                    [
                        "Provide your best answer for the following question. Give ONLY the answer, no other words or explanation.\n",
                        "For example:",
                        "Answer: <most likely answer, as short as possible; not a complete sentence, just the answer!>.",
                    ]
                )
                + "\n\n"
            )
        else:
            prompt = ""

    sample = __format_sample(sample, tokenizer, prompt_style)
    sample.prompt = prompt

    return sample


def get_arc(
    subset=None,
    prompt_style=None,
    train_kshot=0,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    dataset = load_dataset("ai2_arc", subset)
    if not use_cache:
        dataset.cleanup_cache_files()

    train_data, val_data, test_data = [
        data.map(
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
            [train_kshot, eval_kshot, eval_kshot],
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
