import os
import numpy as np

from .registry import register_dataset

__all__ = [
    "get_alpaca_dataset",
]


## HF Convention. See https://huggingface.co/docs/transformers/v4.30.0/en/tasks/token_classification#preprocess.
IGNORE_LABEL = -100


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


def __tokenize_fn(tokenizer, sample):
    final_dict = tokenizer(
        sample["source"] + sample["target"], padding=True, truncation=True
    )

    source_len = len(
        tokenizer(sample["source"], padding=True, truncation=True).input_ids
    )

    labels = np.array(final_dict["input_ids"])
    labels[:source_len] = IGNORE_LABEL
    final_dict["labels"] = labels.tolist()

    return final_dict


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
        lambda x: __tokenize_fn(tokenizer, x),
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
