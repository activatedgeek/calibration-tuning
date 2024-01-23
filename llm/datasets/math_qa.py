import os
import string
import torch

from .registry import register_dataset
from .llm_utils import LMText


__all__ = [
    "get_math_qa",
]


def __format_sample(sample, tokenizer, style):
    target_prompt = "\nAnswer:"

    problem = sample["Problem"]
    answer_map = [opt.split(")")[-1].strip() for opt in sample["options"].split(",")]

    if style == "choice":
        context = "\n".join(
            [
                "Problem:",
                problem,
                "\nChoices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(
                        string.ascii_lowercase[: len(answer_map)], answer_map
                    )
                ],
            ]
        )

        target = sample["correct"] + tokenizer.eos_token
    elif style == "oe":
        context = "\n".join(
            [
                "Provide your best answer for the following math problem. Give ONLY the answer, no other words or explanation.\n"
                "For example:\n",
                "Answer: <most likely answer, as short as possible; not a complete sentence, just the answer!>.\n",
                f"The problem is: {problem}",
            ]
        )

        target = (
            answer_map[string.ascii_lowercase.index(sample["correct"])]
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
            "The following are multiple choice math questions.\n",
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


def get_math_qa(
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
        "math_qa", cache_dir=os.environ.get("HF_DATASETS_CACHE", root)
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    train_data, val_data, test_data = [
        data.map(
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, k, seed=seed
            ).to_pydict(),
            num_proc=num_workers,
            remove_columns=[
                "Problem",
                "Rationale",
                "options",
                "correct",
                "annotated_formula",
                "linear_formula",
                "category",
            ],
        )
        for data, k in zip(
            [dataset.pop("train"), dataset.pop("validation"), dataset.pop("test")],
            [0, eval_kshot, eval_kshot],
        )
    ]

    return train_data, val_data, test_data


@register_dataset(attrs=dict(task_tags=["qa"]))
def math_qa(*args, prompt_style="choice", **kwargs):
    return get_math_qa(
        *args,
        **kwargs,
        prompt_style=prompt_style,
    )
