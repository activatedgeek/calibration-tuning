import os
import string
import torch
import numpy as np

from .registry import register_dataset
from .llm_utils import tokenize_for_causal_lm


__all__ = [
    "get_truthful_qa",
]


def __format_prompt(sample, style, with_answer=False):
    if style == "choice":
        question = sample["question"]
        answer_map = sample["mc1_targets"]["choices"]
        answer = (
            string.ascii_lowercase[np.array(sample["mc1_targets"]["labels"]).argmax()]
            + "</s>\n"
        )

        prompt = "\n".join(
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
                f"Answer: {answer if with_answer else ''}",
            ]
        )

        return prompt

    raise NotImplementedError


def __generate_fewshot_prompts(dataset, prompt_style, kshot, seed=None):
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


def get_truthful_qa(
    root=None,
    subset=None,
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
        "truthful_qa", subset, cache_dir=os.environ.get("HF_DATASETS_CACHE", root)
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    val_data = dataset.pop("validation")
    val_data = val_data.map(
        lambda x: {
            "source": __generate_fewshot_prompts(
                val_data, prompt_style, eval_kshot, seed=seed
            )
            + __format_prompt(x, prompt_style),
            "target": f"{string.ascii_lowercase[np.array(x['mc1_targets']['labels']).argmax()]}{tokenizer.eos_token}",
        },
        num_proc=num_workers,
        remove_columns=[
            "question",
            "mc1_targets",
            "mc2_targets",
        ],
    ).map(
        lambda x: tokenize_for_causal_lm(tokenizer, x),
        num_proc=num_workers,
        remove_columns=["source", "target"],
    )

    return None, val_data, None


@register_dataset
def truthful_qa(*args, **kwargs):
    return get_truthful_qa(
        *args,
        **kwargs,
        subset="multiple_choice",
        prompt_style="choice",
    )
