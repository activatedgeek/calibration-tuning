import string
import numpy as np
from datasets import load_dataset

from ..registry import register_dataset
from ..llm_data_utils import LMText, PromptFormat


def format_sample(sample, style, with_query_label=False, seed=None):
    target_prompt = "\nAnswer:"

    problem = sample["Problem"]
    answer_map = [opt.split(")")[-1].strip() for opt in sample["options"].split(",")]
    target_idx = string.ascii_lowercase.index(sample["correct"])

    output = None
    query_label = (
        np.random.default_rng(seed=seed).binomial(1, 0.5) if with_query_label else None
    )
    output_idx = (
        target_idx
        if query_label == 1
        else (
            np.random.default_rng(seed=seed).choice(
                list(set(range(len(answer_map))) - set([target_idx]))
            )
            if query_label == 0
            else None
        )
    )

    if style == PromptFormat.CHOICE:
        context = "\n".join(
            [
                "Problem:",
                problem,
                "\nChoices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(
                        string.ascii_lowercase[: len(answer_map)], answer_map
                    )
                ],
            ]
        )

        target = string.ascii_lowercase[target_idx]
        if output_idx is not None:
            output = string.ascii_lowercase[output_idx]
    elif style == "oe":
        context = "\n".join(
            [
                "Provide your best answer for the following math problem.",
                f"Problem: {problem}",
            ]
        )

        target = answer_map[target_idx]
        if output_idx is not None:
            output = answer_map[output_idx]
    else:
        raise NotImplementedError(f"Unsupported prompt format {format}.")

    return LMText(
        context=context,
        target_prompt=target_prompt,
        target=target,
        output=output,
        query_label=query_label,
    )


def format_sample_prompt(prompt_dataset, format, kshot=1, seed=None):
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

    if format == PromptFormat.CHOICE:
        prompt = [
            "The following are math questions (with multiple-choice answers).\n",
            *fewshot_samples_prompt,
            "Now, answer the next question.\n\n",
        ]
    elif format == PromptFormat.OE:
        prompt = [
            "The following are math questions (with answers).\n",
            *fewshot_samples_prompt,
            "Now, answer the next question.\n\n",
        ]
    else:
        raise NotImplementedError(f"Unsupported prompt format {format}.")

    return "\n".join(prompt)


def get_math_qa(
    prompt_style=None,
    with_query_label=False,
    train_kshot=0,
    eval_kshot=0,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    format = PromptFormat(prompt_style)

    dataset = load_dataset("math_qa", trust_remote_code=True)
    if not use_cache:
        dataset.cleanup_cache_files()

    dataset = dataset.filter(
        lambda x: len([opt.split(")")[-1].strip() for opt in x["options"].split(",")])
        == len(
            np.unique([opt.split(")")[-1].strip() for opt in x["options"].split(",")])
        ),
        num_proc=num_workers,
    ).map(
        lambda sample, idx: format_sample(
            sample, format, with_query_label=with_query_label, seed=seed + idx
        ).to_pydict(),
        with_indices=True,
        num_proc=num_workers,
        remove_columns=dataset.column_names["test"],
    )

    prompt_data = dataset.get("train")
    prompt_kshot = {
        "train": train_kshot,
        "validation": eval_kshot,
        "test": eval_kshot,
    }

    data_splits = {
        split: ds.map(
            lambda _, idx: {
                "prompt": format_sample_prompt(
                    prompt_data, format, kshot=prompt_kshot[split], seed=seed + idx
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


@register_dataset
def math_qa(*args, **kwargs):
    return get_math_qa(*args, **kwargs)
