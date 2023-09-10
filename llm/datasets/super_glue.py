import os
import string
import torch

from .registry import register_dataset


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

    def __format_sample(sample, tokenizer, style):
        if style == "choice":
            premise = sample["premise"]
            hypothesis = sample["hypothesis"]
            answer_map = ["Yes", "No", "It's impossible to say"]

            source = "\n".join(
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
                    f"Answer: ",
                ]
            )

            target = string.ascii_lowercase[sample["label"]] + tokenizer.eos_token

            return dict(source=source, target=target)

        raise NotImplementedError

    def __generate_fewshot_prompts(
        tokenizer, prompt_style, prompt_dataset, kshot, seed=None
    ):
        if kshot <= 0:
            return ""

        _c = lambda s: s["source"] + s["target"]

        fewshot_prompt = "\n".join(
            [
                "The following are multiple choice questions (with premise, hypothesis, and answers) about entailment.\n",
                *[
                    _c(__format_sample(prompt_dataset[idx], tokenizer, prompt_style))
                    + "\n"
                    for idx in torch.randperm(
                        len(prompt_dataset),
                        generator=torch.Generator().manual_seed(seed),
                    )[:kshot].tolist()
                ],
            ]
        )
        fewshot_prompt = fewshot_prompt + "\nNow, answer the following."

        return fewshot_prompt

    def __format_sample_with_prompt(
        sample, tokenizer, prompt_style, prompt_dataset, kshot, seed=None
    ):
        sample_dict = __format_sample(sample, tokenizer, prompt_style)

        prompt_str = __generate_fewshot_prompts(
            tokenizer, prompt_style, prompt_dataset, kshot, seed=seed
        )
        if len(prompt_str):
            prompt_str += "\n\n"

        sample_dict["source"] = prompt_str + sample_dict["source"]

        return sample_dict

    data_splits = [
        data.filter(lambda x: x["label"] in [0, 1, 2], num_proc=num_workers)
        for data in [
            dataset.pop("train"),
            dataset.pop("validation"),
        ]
    ]
    train_data, val_data = [
        data.map(
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, k, seed=seed
            ),
            num_proc=num_workers,
            remove_columns=[
                "premise",
                "hypothesis",
                "idx",
                "label",
            ],
        )
        for data, k in zip(data_splits, [0, eval_kshot])
    ]

    return train_data, val_data, None


@register_dataset(attrs=dict(task_tags=["entailment"]))
def cb(*args, prompt_style="choice", **kwargs):
    return get_cb(
        *args,
        **kwargs,
        prompt_style=prompt_style,
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

    def __format_sample(sample, tokenizer, style):
        if style == "choice":
            paragraph = sample["paragraph"]
            question = sample["question"]
            answer_map = ["No", "Yes"]

            source = "\n".join(
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
                    f"Answer: ",
                ]
            )

            target = string.ascii_lowercase[sample["label"]] + tokenizer.eos_token

            return dict(source=source, target=target)

        raise NotImplementedError

    def __generate_fewshot_prompts(
        tokenizer, prompt_style, prompt_dataset, kshot, seed=None
    ):
        if kshot <= 0:
            return ""

        _c = lambda s: s["source"] + s["target"]

        fewshot_prompt = "\n".join(
            [
                "The following are reading comprehensions (with answers).\n",
                *[
                    _c(__format_sample(prompt_dataset[idx], tokenizer, prompt_style))
                    + "\n"
                    for idx in torch.randperm(
                        len(prompt_dataset),
                        generator=torch.Generator().manual_seed(seed),
                    )[:kshot].tolist()
                ],
            ]
        )
        fewshot_prompt = fewshot_prompt + "\nNow, answer the following."

        return fewshot_prompt

    def __format_sample_with_prompt(
        sample, tokenizer, prompt_style, prompt_dataset, kshot, seed=None
    ):
        sample_dict = __format_sample(sample, tokenizer, prompt_style)

        prompt_str = __generate_fewshot_prompts(
            tokenizer, prompt_style, prompt_dataset, kshot, seed=seed
        )
        if len(prompt_str):
            prompt_str += "\n\n"

        sample_dict["source"] = prompt_str + sample_dict["source"]

        return sample_dict

    data_splits = [
        data.filter(lambda x: x["label"] in [0, 1, 2], num_proc=num_workers)
        for data in [
            dataset.pop("train"),
            dataset.pop("validation"),
        ]
    ]
    train_data, val_data = [
        data.map(
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, k, seed=seed
            ),
            num_proc=num_workers,
            remove_columns=[
                "paragraph",
                "question",
                "answer",
                "idx",
                "label",
            ],
        )
        for data, k in zip(data_splits, [0, eval_kshot])
    ]

    return train_data, val_data, None


@register_dataset(attrs=dict(task_tags=["comprehension"]))
def multirc(*args, prompt_style="choice", **kwargs):
    return get_multirc(
        *args,
        **kwargs,
        prompt_style=prompt_style,
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

    def __format_sample(sample, tokenizer, style):
        if style == "choice":
            premise = sample["premise"]
            answer_map = [sample["choice1"], sample["choice2"]]

            source = "\n".join(
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
                    f"Answer: ",
                ]
            )

            target = string.ascii_lowercase[sample["label"]] + tokenizer.eos_token

            return dict(source=source, target=target)

        raise NotImplementedError

    def __generate_fewshot_prompts(
        tokenizer, prompt_style, prompt_dataset, kshot, seed=None
    ):
        if kshot <= 0:
            return ""

        _c = lambda s: s["source"] + s["target"]

        fewshot_prompt = "\n".join(
            [
                "The following are reading comprehensions (with answers).\n",
                *[
                    _c(__format_sample(prompt_dataset[idx], tokenizer, prompt_style))
                    + "\n"
                    for idx in torch.randperm(
                        len(prompt_dataset),
                        generator=torch.Generator().manual_seed(seed),
                    )[:kshot].tolist()
                ],
            ]
        )
        fewshot_prompt = fewshot_prompt + "\nNow, answer the following."

        return fewshot_prompt

    def __format_sample_with_prompt(
        sample, tokenizer, prompt_style, prompt_dataset, kshot, seed=None
    ):
        sample_dict = __format_sample(sample, tokenizer, prompt_style)

        prompt_str = __generate_fewshot_prompts(
            tokenizer, prompt_style, prompt_dataset, kshot, seed=seed
        )
        if len(prompt_str):
            prompt_str += "\n\n"

        sample_dict["source"] = prompt_str + sample_dict["source"]

        return sample_dict

    data_splits = [
        data.filter(lambda x: x["label"] in [0, 1, 2], num_proc=num_workers)
        for data in [
            dataset.pop("train"),
            dataset.pop("validation"),
        ]
    ]
    train_data, val_data = [
        data.map(
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, k, seed=seed
            ),
            num_proc=num_workers,
            remove_columns=[
                "premise",
                "choice1",
                "choice2",
                "question",
                "idx",
                "label",
            ],
        )
        for data, k in zip(data_splits, [0, eval_kshot])
    ]

    return train_data, val_data, None


@register_dataset(attrs=dict(task_tags=["commonsense"]))
def copa(*args, prompt_style="choice", **kwargs):
    return get_copa(
        *args,
        **kwargs,
        prompt_style=prompt_style,
    )
