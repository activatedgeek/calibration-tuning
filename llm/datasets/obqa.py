import os
import string
import torch

from .registry import register_dataset


__all__ = [
    "get_openbookqa",
]


def __format_sample(sample, tokenizer, style):
    if style == "choice":
        question = sample["question_stem"]
        answer_map = sample["choices"]["text"]

        source = "\n".join(
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
                f"Answer: ",
            ]
        )

        target = sample["answerKey"].lower() + tokenizer.eos_token

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
            "The following are comprehension passages with multiple choice answers.\n",
            *[
                _c(__format_sample(prompt_dataset[idx], tokenizer, prompt_style)) + "\n"
                for idx in torch.randperm(
                    len(prompt_dataset), generator=torch.Generator().manual_seed(seed)
                )[:kshot].tolist()
            ],
        ]
    )
    fewshot_prompt = (
        fewshot_prompt + "\nNow, answer the next question after the passage."
    )

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


def get_openbookqa(
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
        "openbookqa", cache_dir=os.environ.get("HF_DATASETS_CACHE", root)
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    train_data, val_data, test_data = [
        data.map(
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, k, seed=seed
            ),
            num_proc=num_workers,
            remove_columns=[
                "id",
                "question_stem",
                "choices",
                "answerKey",
            ],
        )
        for data, k in zip(
            [dataset.pop("train"), dataset.pop("validation"), dataset.pop("test")],
            [0, eval_kshot, eval_kshot],
        )
    ]

    return train_data, val_data, test_data


@register_dataset(attrs=dict(task_tags=["qa"]))
def obqa(*args, prompt_style="choice", **kwargs):
    return get_openbookqa(
        *args,
        **kwargs,
        prompt_style=prompt_style,
    )
