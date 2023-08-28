import os
import string
import torch

from .registry import register_dataset
from .llm_utils import tokenize_for_causal_lm


__all__ = [
    "get_trec",
]


def __format_prompt(sample, style, with_answer=False):
    if style == "choice":
        question = sample["text"]
        answer_map = [
            "Abbreviation",
            "Entity",
            "Description and abstract concept",
            "Human being",
            "Location",
            "Numeric value",
        ]
        answer = string.ascii_lowercase[sample["coarse_label"]] + "</s>\n"

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


def get_trec(
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

    dataset = load_dataset("trec", cache_dir=os.environ.get("HF_DATASETS_CACHE", root))
    if not use_cache:
        dataset.cleanup_cache_files()

    train_data, test_data = [
        data.map(
            lambda x: {
                "source": __generate_fewshot_prompts(data, prompt_style, k, seed=seed)
                + __format_prompt(x, prompt_style),
                "target": f"{string.ascii_lowercase[x['coarse_label']]}{tokenizer.eos_token}",
            },
            num_proc=num_workers,
            remove_columns=["text", "coarse_label", "fine_label"],
        ).map(
            lambda x: tokenize_for_causal_lm(tokenizer, x),
            num_proc=num_workers,
            remove_columns=["source", "target"],
        )
        for data, k in zip([dataset.pop("train"), dataset.pop("test")], [0, eval_kshot])
    ]

    return train_data, None, test_data


@register_dataset(attrs=dict(task_tags=["qa"]))
def trec(*args, **kwargs):
    return get_trec(
        *args,
        **kwargs,
        prompt_style="choice",
    )
