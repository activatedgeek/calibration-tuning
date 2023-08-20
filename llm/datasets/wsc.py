import os
import string
import torch

from .registry import register_dataset
from .llm_utils import tokenize_for_causal_lm


__all__ = [
    "get_wsc",
]


def __format_prompt(sample, style, with_answer=False):
    if style == "choice":
        text = sample["text"]
        answer_map = sample["options"]
        answer = string.ascii_lowercase[sample["label"]] + "</s>\n"

        prompt = "\n".join(
            [
                "Sentence:",
                text,
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


def get_wsc(
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
        "winograd_wsc", subset, cache_dir=os.environ.get("HF_DATASETS_CACHE", root)
    )

    test_data = dataset.pop("test")
    test_data = test_data.map(
        lambda x: {
            "source": __generate_fewshot_prompts(
                test_data, prompt_style, kshot=eval_kshot, seed=seed
            )
            + __format_prompt(x, prompt_style),
            "target": f"{string.ascii_lowercase[x['label']]}{tokenizer.eos_token}",
        },
        num_proc=num_workers,
        remove_columns=[
            "text",
            "pronoun",
            "pronoun_loc",
            "quote",
            "quote_loc",
            "options",
            "label",
            "source",
        ],
    ).map(
        lambda x: tokenize_for_causal_lm(tokenizer, x),
        num_proc=num_workers,
        remove_columns=["source", "target"],
    )

    return None, None, test_data


@register_dataset
def wsc(*args, **kwargs):
    return get_wsc(
        *args,
        **kwargs,
        subset="wsc285",
        prompt_style="choice",
    )
