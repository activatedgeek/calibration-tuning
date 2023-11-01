import numpy as np
from datasets import concatenate_datasets

from .registry import get_dataset, list_datasets, register_dataset, get_dataset_attrs


def get_all_train_datasets():
    return sorted(
        list(
            filter(
                lambda x: ("all" not in x) and ("mmlu" not in x) and ("bbh" not in x),
                list_datasets(),
            )
        )
    )


def get_all_eval_datasets():
    return [f"mmlu:{task}" for task in get_dataset_attrs("mmlu").get("tasks")]


def get_combined_train_dataset(
    all_dataset_names,
    max_n=100,
    root=None,
    tokenizer=None,
    seed=None,
    num_workers=8,
    use_dataset_cache=True,
    prompt_style="choice",
    complement=False,
    **_,
):
    all_train_data, all_n = [], []
    for dataset in all_dataset_names:
        train_data, _, _ = get_dataset(
            dataset,
            root=root,
            tokenizer=tokenizer,
            seed=seed,
            num_workers=num_workers,
            use_cache=use_dataset_cache,
            prompt_style=prompt_style,
        )

        if train_data is not None:
            train_data = train_data.shuffle(seed=seed)

            all_train_data.append(train_data)
            all_n.append(len(train_data))

    max_n = min(max_n, sum(all_n))
    select_n = ((np.array(all_n) / sum(all_n)) * max_n).astype(int)

    all_train_data = concatenate_datasets(
        [
            train_data.select(range(n, N) if complement else range(n))
            for train_data, N, n in zip(all_train_data, all_n, select_n)
        ]
    )

    _, a_val_data, a_test_data = get_dataset(
        "mmlu:business_ethics",
        root=root,
        tokenizer=tokenizer,
        seed=seed,
        num_workers=num_workers,
        use_cache=use_dataset_cache,
    )

    return all_train_data, a_val_data, a_test_data


@register_dataset
def all_200k(*args, **kwargs):
    return get_combined_train_dataset(
        all_dataset_names=get_all_train_datasets(),
        *args,
        **kwargs,
        max_n=200_000,
        complement=False,
    )


@register_dataset
def all_200k_c(*args, **kwargs):
    return get_combined_train_dataset(
        all_dataset_names=get_all_train_datasets(),
        *args,
        **kwargs,
        max_n=200_000,
        complement=True,
    )


@register_dataset
def sub_all_200k(*args, **kwargs):
    all_dataset_names = get_all_train_datasets()
    all_dataset_names = all_dataset_names[: len(all_dataset_names) // 2]
    return get_combined_train_dataset(
        all_dataset_names=all_dataset_names,
        *args,
        **kwargs,
        max_n=200_000,
    )


@register_dataset
def sub_all_200k_c(*args, **kwargs):
    all_dataset_names = get_all_train_datasets()
    all_dataset_names = all_dataset_names[len(all_dataset_names) // 2 :]
    return get_combined_train_dataset(
        all_dataset_names=all_dataset_names,
        *args,
        **kwargs,
        max_n=800_000,
    )
