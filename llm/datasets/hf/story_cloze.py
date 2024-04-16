import os
import string
import numpy as np
from datasets import load_dataset

from ..registry import register_dataset
from ..llm_utils import LMText, PromptFormat


def format_sample(sample, format, with_query_label=False, seed=None):
    story = " ".join([sample[f"input_sentence_{i}"] for i in range(1, 5)])
    answer_map = [sample["sentence_quiz1"], sample["sentence_quiz2"]]
    target_idx = sample["answer_right_ending"] - 1

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

    if format == PromptFormat.CHOICE:
        context = "\n".join(
            [
                "Story:",
                story,
                "\nChoices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(
                        string.ascii_lowercase[: len(answer_map)], answer_map
                    )
                ],
            ]
        )

        target_prompt = "\nAnswer:"
        target = string.ascii_lowercase[target_idx]
        if output_idx is not None:
            output = string.ascii_lowercase[output_idx]
    elif format == PromptFormat.OE:
        context = "\n".join(
            [
                "Complete the ending of the following story.",
                story,
            ]
        )

        target_prompt = "\nEnding:"
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
            "The following are stories with endings (multiple-choice).\n",
            *fewshot_samples_prompt,
            "Now, complete the next story.\n\n",
        ]
    elif format == PromptFormat.OE:
        prompt = [
            "The following are stories (with endings).\n",
            *fewshot_samples_prompt,
            "Now, complete the next story.\n\n",
        ]
    else:
        raise NotImplementedError(f"Unsupported prompt format {format}.")

    return "\n".join(prompt)


def get_story_cloze(
    prompt_style=None,
    with_query_label=False,
    eval_kshot=0,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    format = PromptFormat(prompt_style)

    dataset = load_dataset(
        "story_cloze",
        "2018",
        ## NOTE: manually place the CSV in data_dir.
        data_dir=f"{os.environ.get('HF_HOME')}/datasets/story_cloze/2018",
        trust_remote_code=True,
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    dataset = dataset.map(
        lambda sample, idx: format_sample(
            sample, format, with_query_label=with_query_label, seed=seed + idx
        ).to_pydict(),
        with_indices=True,
        num_proc=num_workers,
        remove_columns=dataset.column_names["validation"],
    )
    prompt_data = dataset.get("validation")
    prompt_kshot = {
        "validation": eval_kshot,
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
def story_cloze(*args, **kwargs):
    return get_story_cloze(*args, **kwargs)
