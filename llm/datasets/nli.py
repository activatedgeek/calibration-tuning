import os
import string
import torch

from .registry import register_dataset
from .llm_utils import LMText


__all__ = [
    "get_snli",
    "get_anli",
]


__ATTRS = dict(task_tags=["entailment"])


def __format_sample(sample, tokenizer, style):
    target_prompt = "\nAnswer: "

    if style == "choice":
        premise = sample["premise"]
        hypothesis = sample["hypothesis"]
        answer_map = ["Yes", "It's impossible to say", "No"]

        context = "\n".join(
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
            ]
        )

        target = string.ascii_lowercase[sample["label"]] + tokenizer.eos_token
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
            "The following are multiple choice questions (with premise, hypothesis, and answers) about entailment.\n",
            *[
                str(__format_sample(prompt_dataset[idx], tokenizer, prompt_style))
                + "\n"
                for idx in torch.randperm(
                    len(prompt_dataset), generator=torch.Generator().manual_seed(seed)
                )[:kshot].tolist()
            ],
        ]
    )
    fewshot_prompt = fewshot_prompt + "\nNow, answer the following."

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


def get_snli(
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

    dataset = load_dataset("snli", cache_dir=os.environ.get("HF_DATASETS_CACHE", root))
    if not use_cache:
        dataset.cleanup_cache_files()

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
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, k, seed=seed
            ).to_pydict(),
            num_proc=num_workers,
            remove_columns=[
                "premise",
                "hypothesis",
                "label",
            ],
        )
        for data, k in zip(data_splits, [0, eval_kshot, eval_kshot])
    ]

    return train_data, val_data, test_data


@register_dataset(attrs=__ATTRS)
def snli(*args, prompt_style="choice", **kwargs):
    return get_snli(
        *args,
        **kwargs,
        prompt_style=prompt_style,
    )


def get_anli(
    root=None,
    round=None,
    prompt_style=None,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    from datasets import load_dataset

    assert round in [1, 2, 3], "Invalid round value"

    dataset = load_dataset("anli", cache_dir=os.environ.get("HF_DATASETS_CACHE", root))
    if not use_cache:
        dataset.cleanup_cache_files()

    dev_data = dataset.pop(f"dev_r{round}")

    data_splits = [
        data.filter(lambda x: x["label"] in [0, 1, 2], num_proc=num_workers)
        for data in [dataset.pop(f"train_r{round}"), dataset.pop(f"test_r{round}")]
    ]
    train_data, test_data = [
        data.map(
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, k, seed=seed
            ).to_pydict(),
            num_proc=num_workers,
            remove_columns=[
                "uid",
                "premise",
                "hypothesis",
                "label",
                "reason",
            ],
        )
        for data, k in zip(data_splits, [0, eval_kshot])
    ]

    return train_data, None, test_data


@register_dataset(attrs=__ATTRS)
def anli_r1(*args, prompt_style="choice", **kwargs):
    return get_anli(
        *args,
        **kwargs,
        round=1,
        prompt_style=prompt_style,
    )


@register_dataset(attrs=__ATTRS)
def anli_r2(*args, prompt_style="choice", **kwargs):
    return get_anli(
        *args,
        **kwargs,
        round=2,
        prompt_style=prompt_style,
    )


@register_dataset(attrs=__ATTRS)
def anli_r3(*args, prompt_style="choice", **kwargs):
    return get_anli(
        *args,
        **kwargs,
        round=3,
        prompt_style=prompt_style,
    )
