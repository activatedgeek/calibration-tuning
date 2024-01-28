import os
import string
import torch

from .registry import register_dataset
from .llm_utils import LMText


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
        "super_glue",
        "cb",
        cache_dir=os.environ.get("HF_DATASETS_CACHE", root),
        trust_remote_code=True,
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    def __format_sample(sample, tokenizer, style):
        target_prompt = "\nAnswer:"

        premise = sample["premise"]
        hypothesis = sample["hypothesis"]
        answer_map = ["Yes", "No", "It's impossible to say"]

        if style == "choice":
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

            target = string.ascii_lowercase[sample["label"]]
        elif style == "oe":
            context = "\n".join(
                [
                    "Read the following premise and answer if the hypothesis is true.",
                    premise,
                    f'Hypothesis: {hypothesis}. Is the answer "Yes", "No", or "It\'s impossible to say"? Respond with only the answer and no additional text.',
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
            return ""

        fewshot_prompt = "\n".join(
            [
                "The following are multiple choice questions (with premise, hypothesis, and answers) about entailment.\n",
                *[
                    str(__format_sample(prompt_dataset[idx], tokenizer, prompt_style))
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
        prompt = __generate_fewshot_prompts(
            tokenizer, prompt_style, prompt_dataset, kshot, seed=seed
        )
        if len(prompt):
            prompt += "\n\n"

        sample = __format_sample(sample, tokenizer, prompt_style)
        sample.prompt = prompt

        return sample

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
            ).to_pydict(),
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
        "super_glue",
        "multirc",
        cache_dir=os.environ.get("HF_DATASETS_CACHE", root),
        trust_remote_code=True,
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    def __format_sample(sample, tokenizer, style):
        target_prompt = "\nAnswer:"

        paragraph = sample["paragraph"]
        question = sample["question"]
        answer = sample["answer"]
        answer_map = ["No", "Yes"]

        if style == "choice":
            context = "\n".join(
                [
                    "Paragraph:",
                    paragraph,
                    f"\nQ: {question}",
                    f"\nA: {answer}" "\n\nChoices:",
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
                    "Read the following paragraph along with the question and answer. Then, respond with whether the answer is correct.",
                    f"Passage: {paragraph}\n",
                    f"Question: {question}\n",
                    f"Answer: {answer}\n",
                    'Is the answer correct? Respond with only "Yes" or "No" and no additional text.',
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
            return ""

        fewshot_prompt = "\n".join(
            [
                "The following are reading comprehensions (with answers).\n",
                *[
                    str(__format_sample(prompt_dataset[idx], tokenizer, prompt_style))
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
        prompt = __generate_fewshot_prompts(
            tokenizer, prompt_style, prompt_dataset, kshot, seed=seed
        )
        if len(prompt):
            prompt += "\n\n"

        sample = __format_sample(sample, tokenizer, prompt_style)
        sample.prompt = prompt

        return sample

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
            ).to_pydict(),
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
        "super_glue",
        "copa",
        cache_dir=os.environ.get("HF_DATASETS_CACHE", root),
        trust_remote_code=True,
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    def __format_sample(sample, tokenizer, style):
        target_prompt = "\nAnswer:"

        premise = sample["premise"]
        answer_map = [sample["choice1"], sample["choice2"]]

        if style == "choice":
            context = "\n".join(
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
                ]
            )

            target = string.ascii_lowercase[sample["label"]]
        elif style == "oe":
            context = "\n".join(
                [
                    "Read the following premise and pick the correct choice. Then, respond with which of the choices is correct.",
                    f"Premise: {premise}\n",
                    f"Choice 1: {answer_map[0]}",
                    f"Choice 2: {answer_map[1]}",
                    'Which of the choices is correct? Respond with only "1" or "2" and no additional text.',
                ]
            )

            target = str(sample["label"] + 1)
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
                "The following are reading comprehensions (with answers).\n",
                *[
                    str(__format_sample(prompt_dataset[idx], tokenizer, prompt_style))
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
        prompt = __generate_fewshot_prompts(
            tokenizer, prompt_style, prompt_dataset, kshot, seed=seed
        )
        if len(prompt):
            prompt += "\n\n"

        sample = __format_sample(sample, tokenizer, prompt_style)
        sample.prompt = prompt

        return sample

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
            ).to_pydict(),
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
