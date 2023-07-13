import os

from .registry import register_dataset
from .llm_utils import tokenize_for_causal_lm

__all__ = [
    "get_alpaca_dataset",
]


def __format_prompt(sample, style):
    if style == "v1":
        instruction = sample["instruction"]
        raw_input = sample.get("input", "")
        _prompt_input = (
            ""
            if raw_input == ""
            else ", paired with an input that provides further context"
        )
        _input = "" if raw_input == "" else f"### Input:\n{raw_input}\n\n"

        prompt = (
            f"Below is an instruction that describes a task{_prompt_input}.\n\n"
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"{_input}"
            "### Response:"
        )

        return prompt

    raise NotImplementedError


def get_alpaca_dataset(
    root=None,
    seed=None,
    tokenizer=None,
    prompt_style=None,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "tatsu-lab/alpaca", cache_dir=os.environ.get("DATADIR", root)
    )

    dataset = dataset.map(
        lambda x: {
            "source": __format_prompt(x, prompt_style),
            "target": f"{x['output']}{tokenizer.eos_token}",
        },
        num_proc=4,
        remove_columns=[
            "instruction",
            "input",
            "output",
            "text",
        ],
    ).map(
        lambda x: tokenize_for_causal_lm(tokenizer, x),
        num_proc=4,
        remove_columns=[
            "source",
            "target",
        ],
    )

    train_data = dataset["train"].shuffle(seed=seed)

    return train_data, None, None


@register_dataset
def alpaca(*args, **kwargs):
    return get_alpaca_dataset(
        *args,
        **kwargs,
        prompt_style="v1",
    )
