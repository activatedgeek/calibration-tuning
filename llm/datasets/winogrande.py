import os
import string
import torch

from .registry import register_dataset
from .llm_utils import tokenize_for_causal_lm


__all__ = [
    "get_winogrande",
]


def __format_prompt(sample, style, with_answer=False):
    if style == "choice":
        sentence = sample["sentence"]
        answer_map = [sample["option1"], sample["option2"]]
        answer = string.ascii_lowercase[int(sample["answer"]) - 1] + "</s>\n"

        prompt = "\n".join(
            [
                "Sentence:",
                sentence,
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


def __generate_fewshot_prompts(dataset, prompt_style, kshot=5, seed=None):
    if kshot <= 0:
        return ""

    fewshot_prompt = "\n".join(
        [
            "The following are sentences with ambiguity.\n",
            *[
                __format_prompt(dataset[idx], prompt_style, with_answer=True)
                for idx in torch.randperm(
                    len(dataset), generator=torch.Generator().manual_seed(seed)
                )[:kshot].tolist()
            ],
        ]
    )
    fewshot_prompt = fewshot_prompt + "\nNow, resolve the next ambiguity.\n\n"

    return fewshot_prompt


def get_winogrande(
    root=None,
    subset=None,
    prompt_style=None,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "winogrande", subset, cache_dir=os.environ.get("HF_DATASETS_CACHE", root)
    )

    data_splits = [
        data.filter(lambda x: x["answer"] in ["1", "2"])
        for data in [
            dataset.pop("train"),
            dataset.pop("validation"),
            dataset.pop("test"),
        ]
    ]
    train_data, val_data, test_data = [
        data.map(
            lambda x: {
                "source": __generate_fewshot_prompts(
                    data, prompt_style, kshot=k, seed=seed
                )
                + __format_prompt(x, prompt_style),
                "target": f"{string.ascii_lowercase[int(x['answer']) - 1]}{tokenizer.eos_token}",
            },
            num_proc=num_workers,
            remove_columns=[
                "sentence",
                "option1",
                "option2",
                "answer",
            ],
        ).map(
            lambda x: tokenize_for_causal_lm(tokenizer, x),
            num_proc=num_workers,
            remove_columns=["source", "target"],
        )
        for data, k in zip(data_splits, [0, eval_kshot, eval_kshot])
    ]

    return train_data, val_data, test_data


@register_dataset
def winogrande(*args, **kwargs):
    return get_winogrande(
        *args,
        **kwargs,
        subset="winogrande_xl",
        prompt_style="choice",
    )
