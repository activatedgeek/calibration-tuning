import os
import string
import torch
import numpy as np

from .registry import register_dataset
from .llm_utils import LMText


__all__ = [
    "get_truthful_qa",
]


def __format_sample(sample, tokenizer, style):
    target_prompt = "\nAnswer:"

    question = sample["question"]
    answer_map = sample["mc1_targets"]["choices"]

    if style == "choice":
        context = "\n".join(
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
            ]
        )

        target = string.ascii_lowercase[
            np.array(sample["mc1_targets"]["labels"]).argmax()
        ]
    elif style == "oe":
        context = "\n".join(
            [
                f"Question: {question}",
            ]
        )

        target = answer_map[np.array(sample["mc1_targets"]["labels"]).argmax()]
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
        lambda x: __format_sample_with_prompt(
            x, tokenizer, prompt_style, val_data, eval_kshot, seed=seed
        ).to_pydict(),
        num_proc=num_workers,
        remove_columns=[
            "question",
            "mc1_targets",
            "mc2_targets",
        ],
    )

    return None, val_data, None


@register_dataset
def truthful_qa(*args, prompt_style="choice", **kwargs):
    return get_truthful_qa(
        *args,
        **kwargs,
        subset="multiple_choice",
        prompt_style=prompt_style,
    )
