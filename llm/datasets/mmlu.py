import os
import string

from .registry import register_dataset

__all__ = [
    "get_mmlu",
]


def __format_prompt(sample, style):
    if style == "mcq":
        question = sample["question"]
        choices = sample["choices"]

        prompt = "\n".join(
            [
                f"Question: {question}",
                "Choices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(string.ascii_uppercase[: len(choices)], choices)
                ],
                "Answer: ",
            ]
        )

        return prompt

    raise NotImplementedError


def get_mmlu(
    root=None,
    instance=None,
    prompt_style=None,
    seed=None,
    tokenizer=None,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "cais/mmlu", instance, cache_dir=os.environ.get("DATADIR", root)
    )

    dataset = dataset.map(
        lambda x: {
            "prompt": __format_prompt(x, prompt_style),
            "label": x["answer"],
        },
        remove_columns=[
            "question",
            "choices",
            "answer",
        ],
        num_proc=4,
    ).map(
        lambda x: tokenizer(x["prompt"], padding=True, truncation=True),
        batched=True,
        num_proc=4,
        remove_columns=["prompt"],
    )

    dev_data = dataset["dev"].shuffle(seed=seed)
    val_data = dataset["validation"].shuffle(seed=seed)
    test_data = dataset["test"].shuffle(seed=seed)

    return dev_data, val_data, test_data


@register_dataset
def mmlu(*args, instance=None, prompt_style="mcq", **kwargs):
    return get_mmlu(
        *args,
        **kwargs,
        instance=instance,
        prompt_style=prompt_style,
    )
