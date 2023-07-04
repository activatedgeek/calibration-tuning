import os
import string

from .registry import register_dataset

__all__ = [
    "get_mmlu",
]

__ATTRS = dict(label2char=lambda idx: string.ascii_lowercase[idx])


## TODO: add few-shot prompts.
def __format_prompt(sample, style, with_answer=False):
    if style == "mcq":
        question = sample["question"]
        choices = sample["choices"]
        answer = string.ascii_lowercase[sample["answer"]] + "\n"

        prompt = "\n".join(
            [
                f"Question: {question}",
                "Choices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(string.ascii_lowercase[: len(choices)], choices)
                ],
                f"Answer: {answer if with_answer else ''}",
            ]
        )

        return prompt

    raise NotImplementedError


def get_mmlu(
    root=None,
    instance=None,
    prompt_style=None,
    kshot=1,
    seed=None,
    tokenizer=None,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "cais/mmlu", instance, cache_dir=os.environ.get("DATADIR", root)
    )

    if kshot:
        fewshot_prompt = "\n".join(
            [
                f"The following are multiple choice questions (with answers) about {' '.join(instance.split('_'))}.\n",
                *[
                    __format_prompt(dataset["dev"][idx], prompt_style, with_answer=True)
                    for idx in range(kshot)
                ],
                "Now please answer the following question with the correct choice letter only.\n",
            ]
        )
    else:
        fewshot_prompt = ""

    dataset = dataset.map(
        lambda x: {
            "prompt": fewshot_prompt + __format_prompt(x, prompt_style),
            "label": x["answer"],
        },
        remove_columns=[
            "question",
            "choices",
            "answer",
        ],
        num_proc=4,
    ).map(
        lambda x: tokenizer(x["prompt"], padding=True),
        batched=True,
        num_proc=4,
        remove_columns=["prompt"],
    )

    dev_data = dataset["dev"].shuffle(seed=seed)
    val_data = dataset["validation"].shuffle(seed=seed)
    test_data = dataset["test"].shuffle(seed=seed)

    return dev_data, val_data, test_data


@register_dataset(attrs=__ATTRS)
def mmlu(*args, instance=None, prompt_style="mcq", **kwargs):
    return get_mmlu(
        *args,
        **kwargs,
        instance=instance,
        prompt_style=prompt_style,
    )
