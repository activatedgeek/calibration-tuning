import os
import string

from .registry import register_dataset

__all__ = [
    "get_truthful_qa",
]

__ATTRS = dict(label2char=lambda idx: string.ascii_lowercase[idx])


## TODO: add few-shot prompts.
def __format_prompt(sample, style):
    if style == "mcq":
        question = sample["question"]
        choices = sample["mc1_targets.choices"]

        prompt = "\n".join(
            [
                f"Question: {question}",
                "Choices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(string.ascii_lowercase[: len(choices)], choices)
                ],
                "Answer: ",
            ]
        )

        return prompt

    raise NotImplementedError


def get_truthful_qa(
    root=None,
    instance=None,
    mode=None,
    prompt_style=None,
    seed=None,
    tokenizer=None,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "truthful_qa", instance, cache_dir=os.environ.get("DATADIR", root)
    ).flatten()

    if mode == "mc1":
        dataset = dataset.map(
            lambda x: {
                "prompt": __format_prompt(x, prompt_style),
                "label": x["mc1_targets.labels"].index(1),
            },
            remove_columns=[
                "question",
                "mc1_targets.choices",
                "mc1_targets.labels",
                "mc2_targets.choices",
                "mc2_targets.labels",
            ],
            num_proc=4,
        ).map(
            lambda x: tokenizer(x["prompt"], padding=True),
            batched=True,
            num_proc=4,
            remove_columns=["prompt"],
        )
    else:
        raise NotImplementedError

    val_data = dataset["validation"].shuffle(seed=seed)

    return None, val_data, None


## NOTE: Some arguments ignored to avoid conflict.
@register_dataset(attrs=__ATTRS)
def truthful_qa_mc1(*args, instance=None, prompt_style="mcq", **kwargs):
    return get_truthful_qa(
        *args,
        **kwargs,
        instance="multiple_choice",
        mode="mc1",
        prompt_style=prompt_style,
    )
