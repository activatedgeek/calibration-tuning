import os
import string
import torch

from .registry import register_dataset
from .llm_utils import tokenize_for_causal_lm


__all__ = [
    "get_piqa",
]


def __format_prompt(sample, style, with_answer=False):
    if style == "choice":
        goal = sample["goal"]
        answer_map = [sample["sol1"], sample["sol2"]]
        assert sample["label"] in [0, 1], sample
        answer = string.ascii_lowercase[sample["label"]] + "</s>\n"

        prompt = "\n".join(
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
            "The following are multiple choice questions.\n",
            *[
                __format_prompt(dataset[idx], prompt_style, with_answer=True)
                for idx in torch.randperm(
                    len(dataset), generator=torch.Generator().manual_seed(seed)
                )[:kshot].tolist()
            ],
        ]
    )
    fewshot_prompt = fewshot_prompt + "\nNow, answer the next question.\n\n"

    return fewshot_prompt


def get_piqa(
    root=None,
    prompt_style=None,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset("piqa", cache_dir=os.environ.get("HF_DATASETS_CACHE", root))
    dataset.cleanup_cache_files()

    ## NOTE: "test" split has no labels.
    train_data, val_data = [
        data.filter(lambda x: x["label"] in [0, 1], num_proc=num_workers).map(
            lambda x: {
                "source": __generate_fewshot_prompts(
                    data, prompt_style, kshot=k, seed=seed
                )
                + __format_prompt(x, prompt_style),
                "target": f"{string.ascii_lowercase[x['label']]}{tokenizer.eos_token}",
            },
            num_proc=num_workers,
            remove_columns=[
                "goal",
                "sol1",
                "sol2",
                "label",
            ],
        )
        .map(
            lambda x: tokenize_for_causal_lm(tokenizer, x),
            num_proc=num_workers,
            remove_columns=["source", "target"],
        )
        for data, k in zip(
            [dataset.pop("train"), dataset.pop("validation")],
            [0, eval_kshot],
        )
    ]

    return train_data, val_data, None


@register_dataset
def piqa(*args, **kwargs):
    return get_piqa(
        *args,
        **kwargs,
        prompt_style="choice",
    )
