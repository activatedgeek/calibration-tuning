from datasets import load_dataset, DatasetDict

from ..registry import register_dataset
from ..llm_data_utils import PromptFormat
from .snli import format_sample, format_sample_prompt


def get_anli(
    round=None,
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

    dataset = load_dataset("anli")
    if not use_cache:
        dataset.cleanup_cache_files()

    dataset = DatasetDict(
        {k.split("_")[0]: v for k, v in dataset.items() if k.endswith(f"_r{round}")}
    )

    dataset = dataset.filter(
        lambda x: x["label"] in [0, 1, 2], num_proc=num_workers
    ).map(
        lambda sample, idx: format_sample(
            sample, format, with_query_label=with_query_label, seed=seed + idx
        ).to_pydict(),
        with_indices=True,
        num_proc=num_workers,
        remove_columns=dataset.column_names["test"],
    )

    prompt_data = dataset.pop("dev")
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
def anli_r1(*args, **kwargs):
    return get_anli(*args, **kwargs, round=1)


@register_dataset
def anli_r2(*args, **kwargs):
    return get_anli(*args, **kwargs, round=2)


@register_dataset
def anli_r3(*args, **kwargs):
    return get_anli(*args, **kwargs, round=3)
