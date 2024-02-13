import os
import string
import torch

from .registry import register_dataset
from .llm_utils import LMText


__all__ = [
    "get_trec",
]


def __format_sample(sample, tokenizer, style):
    target_prompt = "\nAnswer:"

    question = sample["text"]
    answer_map = [
        "Abbreviation",
        "Entity",
        "Description and abstract concept",
        "Human being",
        "Location",
        "Numeric value",
    ]

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

        target = string.ascii_lowercase[sample["coarse_label"]]
    elif style == "oe":
        context = "\n".join(
            [
                "Read the following question and then pick a category that describes the question.",
                f"Question: {question}",
                f"What category is the question in? Choose one from {', '.join(answer_map)}. Respond with only the category and no additional text.",
            ]
        )

        target = answer_map[sample["coarse_label"]]
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
            "The following are multiple choice questions.\n",
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


def get_trec(
    root=None,
    prompt_style=None,
    train_kshot=0,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "trec",
        cache_dir=os.environ.get("HF_DATASETS_CACHE", root),
        trust_remote_code=True,
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    train_data, test_data = [
        data.map(
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, k, seed=seed
            ).to_pydict(),
            num_proc=num_workers,
            remove_columns=["text", "coarse_label", "fine_label"],
        )
        for data, k in zip(
            [dataset.pop("train"), dataset.pop("test")], [train_kshot, eval_kshot]
        )
    ]

    return train_data, None, test_data


@register_dataset(attrs=dict(task_tags=["qa"]))
def trec(*args, prompt_style="choice", **kwargs):
    return get_trec(
        *args,
        **kwargs,
        prompt_style=prompt_style,
    )
