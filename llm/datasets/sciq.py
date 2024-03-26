import string
import torch
import numpy as np
from datasets import load_dataset

from .registry import register_dataset
from .llm_utils import LMText


__all__ = [
    "get_sciq",
]


def __format_sample(sample, tokenizer, style):
    target_prompt = "\nAnswer:"

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

    if style == "choice":
        context = "\n".join(
            (["Support:", support] if len(support) else [])
            + [
                f"\nQuestion:\n{question}",
                "\nChoices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(
                        string.ascii_lowercase[: len(answer_map)], answer_map
                    )
                ],
            ]
        )

        target = string.ascii_lowercase[answer_order.index(3)]
    elif style == "oe":
        context = "\n".join(
            [
                "Read the following and answer the question.",
            ]
            + ([support + "\n"] if len(support) else [])
            + [
                f"Question: {question}",
            ]
        )

        target = answer_map[answer_order.index(3)]
    else:
        raise NotImplementedError

    return LMText(context=context, target_prompt=target_prompt, target=target)


def __generate_fewshot_prompts(
    tokenizer, prompt_style, prompt_dataset, kshot, seed=None
):
    if kshot <= 0:
        if prompt_style == "oe":
            return "\n".join(
                [
                    "Give ONLY the answer, no other words or explanation.\n",
                    "For example:",
                    "Answer: <most likely answer, as short as possible>.",
                ]
            )
        return ""

    fewshot_prompt = "\n".join(
        [
            "The following are science exam questions with answers.\n",
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


def get_sciq(
    prompt_style=None,
    train_kshot=0,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    dataset = load_dataset("sciq")
    if not use_cache:
        dataset.cleanup_cache_files()

    train_data, val_data, test_data = [
        data.map(
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, k, seed=seed
            ).to_pydict(),
            num_proc=num_workers,
            remove_columns=[
                "question",
                "distractor1",
                "distractor2",
                "distractor3",
                "correct_answer",
                "support",
            ],
        )
        for data, k in zip(
            [dataset.pop("train"), dataset.pop("validation"), dataset.pop("test")],
            [train_kshot, eval_kshot, eval_kshot],
        )
    ]

    return train_data, val_data, test_data


@register_dataset(attrs=dict(task_tags=["qa"]))
def sciq(*args, prompt_style="choice", **kwargs):
    return get_sciq(
        *args,
        **kwargs,
        prompt_style=prompt_style,
    )
