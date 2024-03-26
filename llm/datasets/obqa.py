import string
import torch
from datasets import load_dataset

from .registry import register_dataset
from .llm_utils import LMText


__all__ = [
    "get_openbookqa",
]


def __format_sample(sample, tokenizer, style):
    target_prompt = "\nAnswer:"

    question = sample["question_stem"]
    answer_map = sample["choices"]["text"]

    if style == "choice":
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

        target = sample["answerKey"].lower()
    elif style == "oe":
        context = "\n".join(
            [
                "Provide your best answer for the following question.",
                f"The question is: {question}",
            ]
        )

        target = answer_map[string.ascii_lowercase.index(sample["answerKey"].lower())]
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
            "The following are comprehension passages with answers.\n",
            *[
                str(__format_sample(prompt_dataset[idx], tokenizer, prompt_style))
                + "\n"
                for idx in torch.randperm(
                    len(prompt_dataset), generator=torch.Generator().manual_seed(seed)
                )[:kshot].tolist()
            ],
        ]
    )
    fewshot_prompt = (
        fewshot_prompt + "\nNow, answer the next question after the passage."
    )

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
                        "Give ONLY the answer, no other words or explanation.\n",
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


def get_openbookqa(
    prompt_style=None,
    train_kshot=0,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    dataset = load_dataset("openbookqa")
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
                "question_stem",
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
def obqa(*args, prompt_style="choice", **kwargs):
    return get_openbookqa(
        *args,
        **kwargs,
        prompt_style=prompt_style,
    )
