import os
import string
import torch

from .registry import register_dataset
from .llm_utils import LMText


__all__ = [
    "get_winogrande",
]


def __format_sample(sample, tokenizer, style):
    target_prompt = "\nAnswer: "

    sentence = sample["sentence"]

    if style == "choice":
        answer_map = [sample["option1"], sample["option2"]]

        context = "\n".join(
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
            ]
        )

        target = string.ascii_lowercase[int(sample["answer"]) - 1] + tokenizer.eos_token
    elif style == "oe":
        blank_idx = sentence.index("_")
        answer_map = [
            "".join(
                [sentence[:blank_idx], sample["option1"], sentence[blank_idx + 1 :]]
            ),
            "".join(
                [sentence[:blank_idx], sample["option2"], sentence[blank_idx + 1 :]]
            ),
        ]

        context = "\n".join(
            [
                "Fill in the blank (_) in the following sentence from the following choices.",
                f"Sentence: {sentence}\n",
                f"Choice 1: {answer_map[0]}",
                f"Choice 2: {answer_map[1]}",
                'Which of the choices is correct? Respond with only "1" or "2" and no additional text.',
            ]
        )

        target = sample["answer"] + tokenizer.eos_token
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
            "The following are sentences with ambiguity.\n",
            *[
                str(__format_sample(prompt_dataset[idx], tokenizer, prompt_style))
                + "\n"
                for idx in torch.randperm(
                    len(prompt_dataset), generator=torch.Generator().manual_seed(seed)
                )[:kshot].tolist()
            ],
        ]
    )
    fewshot_prompt = fewshot_prompt + "\nNow, resolve the next ambiguity."

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


def get_winogrande(
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
        "winogrande", subset, cache_dir=os.environ.get("HF_DATASETS_CACHE", root)
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    data_splits = [
        data.filter(lambda x: x["answer"] in ["1", "2"])
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
                "sentence",
                "option1",
                "option2",
                "answer",
            ],
        )
        for data, k in zip(data_splits, [0, eval_kshot])
    ]

    return train_data, val_data, None


@register_dataset(attrs=dict(task_tags=["coreference"]))
def winogrande(*args, prompt_style="choice", **kwargs):
    return get_winogrande(
        *args,
        **kwargs,
        subset="winogrande_xl",
        prompt_style=prompt_style,
    )
