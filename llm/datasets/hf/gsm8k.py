import numpy as np
from datasets import load_dataset

from ..registry import register_dataset
from ..llm_data_utils import LMText, PromptFormat


def format_sample(sample, format):
    target_prompt = "\nAnswer:"

    question = sample["question"]

    if format == PromptFormat.OE:
        context = "\n".join([f"Question: {question}"])

        target = sample["answer"]
    else:
        raise NotImplementedError(f"Unsupported prompt format {format}.")

    return LMText(context=context, target_prompt=target_prompt, target=target)


def format_sample_prompt(prompt_dataset, format, kshot=8, seed=None):
    if not kshot:
        return ""

    samples_idx = (
        np.random.default_rng(seed=seed)
        .permutation(len(prompt_dataset))[:kshot]
        .tolist()
    )

    fewshot_samples_prompt = [
        str(LMText.from_(prompt_dataset[idx])) + "\n" for idx in samples_idx
    ]

    if format == PromptFormat.OE:
        prompt = [
            "The following are math questions (with answers).\n",
            *fewshot_samples_prompt,
            "Now, answer the next question.\n\n",
        ]
    else:
        raise NotImplementedError(f"Unsupported prompt format {format}.")

    return "\n".join(prompt)


def get_gsm8k(
    prompt_style=None,
    train_kshot=0,
    eval_kshot=8,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    format = PromptFormat(prompt_style)

    dataset = load_dataset("gsm8k", "main")
    if not use_cache:
        dataset.cleanup_cache_files()

    dataset = dataset.map(
        lambda sample: format_sample(sample, format).to_pydict(),
        num_proc=num_workers,
        remove_columns=dataset.column_names["test"],
    )

    prompt_data = dataset.get("train")
    prompt_kshot = {
        "train": train_kshot,
        "test": eval_kshot,
    }

    data_splits = {
        split: ds.map(
            lambda _, idx: {
                "prompt": format_sample_prompt(
                    prompt_data,
                    format,
                    kshot=prompt_kshot[split],
                    seed=seed + idx,
                )
            },
            with_indices=True,
            num_proc=num_workers,
        )
        for split, ds in dataset.items()
    }

    train_data = data_splits.pop("train", None)
    val_data = data_splits.pop("validation", None)
    test_data = data_splits.pop("test", None)

    return train_data, val_data, test_data


@register_dataset(attrs=dict(eval=True))
def gsm8k(*args, **kwargs):
    return get_gsm8k(*args, **kwargs)
