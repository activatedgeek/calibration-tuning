import string
import torch
from datasets import load_dataset

from .registry import register_dataset
from .llm_utils import LMText


__all__ = [
    "get_piqa",
]


def __format_sample(sample, tokenizer, style):
    target_prompt = "\nAnswer:"

    goal = sample["goal"]
    answer_map = [sample["sol1"], sample["sol2"]]

    if style == "choice":
        context = "\n".join(
            [
                "Goal:",
                goal,
                "\nChoices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(
                        string.ascii_lowercase[: len(answer_map)], answer_map
                    )
                ],
            ]
        )

        target = string.ascii_lowercase[sample["label"]]
    elif style == "oe":
        context = "\n".join(
            [
                "Provide advice on how to accomplish the following goal.\n",
                f"Goal: {goal}",
            ]
        )

        target = answer_map[sample["label"]]
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
                    "Give ONLY the advice, no other words or explanation.\n",
                    "For example:",
                    "Answer: <advice, as short as possible; a single sentence!>.",
                ]
            )

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
    prompt = __generate_fewshot_prompts(
        tokenizer, prompt_style, prompt_dataset, kshot, seed=seed
    )
    if len(prompt):
        prompt += "\n\n"

    sample = __format_sample(sample, tokenizer, prompt_style)
    sample.prompt = prompt

    return sample


def get_piqa(
    prompt_style=None,
    train_kshot=0,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    dataset = load_dataset("piqa")
    if not use_cache:
        dataset.cleanup_cache_files()

    ## NOTE: "test" split has no labels.
    data_splits = [
        data.filter(lambda x: x["label"] in [0, 1, 2], num_proc=num_workers)
        for data in [dataset.pop("train"), dataset.pop("validation")]
    ]
    train_data, val_data = [
        data.map(
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, k, seed=seed
            ).to_pydict(),
            num_proc=num_workers,
            remove_columns=[
                "goal",
                "sol1",
                "sol2",
                "label",
            ],
        )
        for data, k in zip(data_splits, [train_kshot, eval_kshot])
    ]

    return train_data, val_data, None


@register_dataset(attrs=dict(task_tags=["qa", "commonsense"]))
def piqa(*args, prompt_style="choice", **kwargs):
    return get_piqa(
        *args,
        **kwargs,
        prompt_style=prompt_style,
    )
