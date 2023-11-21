import numpy as np
from datasets import concatenate_datasets

from .registry import get_dataset, list_datasets, register_dataset, get_dataset_attrs


def get_all_datasets_list(dataset_str):
    dataset, sub_dataset = dataset_str.split(":")

    assert dataset in [
        "all",
        "eval",
    ], f"Format strings as <all|eval>:<split>, found {dataset_str}"

    all_datasets_list = []

    if dataset == "all":
        if sub_dataset == "train":
            all_datasets_list += sorted(
                list(
                    filter(
                        lambda x: not any(
                            s in x for s in ["all", "sub", "mmlu", "bbmc"]
                        ),
                        list_datasets(),
                    )
                )
            )
        else:
            raise NotImplementedError
    elif dataset == "eval":
        mmlu_tasks = [f"mmlu:{task}" for task in get_dataset_attrs("mmlu").get("tasks")]
        bbmc_tasks = [f"bbmc:{task}" for task in get_dataset_attrs("bbmc").get("tasks")]

        if sub_dataset == "all":
            all_datasets_list += mmlu_tasks + bbmc_tasks
        elif sub_dataset == "mmlu":
            all_datasets_list += mmlu_tasks
        elif sub_dataset == "bbmc":
            all_datasets_list += bbmc_tasks
        else:
            raise NotImplementedError

    return all_datasets_list


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
        all_dataset_names=get_all_datasets_list("all:train"),
        *args,
        **kwargs,
        max_n=200_000,
        complement=False,
    )


@register_dataset
def all_200k_c(*args, **kwargs):
    return get_combined_train_dataset(
        all_dataset_names=get_all_datasets_list("all:train"),
        *args,
        **kwargs,
        max_n=200_000,
        complement=True,
    )


@register_dataset
def sub_200k(*args, **kwargs):
    all_dataset_names = get_all_datasets_list("all:train")
    all_dataset_names = all_dataset_names[: len(all_dataset_names) // 2]
    return get_combined_train_dataset(
        all_dataset_names=all_dataset_names,
        *args,
        **kwargs,
        max_n=200_000,
    )


@register_dataset
def sub_200k_c(*args, **kwargs):
    all_dataset_names = get_all_datasets_list("all:train")
    all_dataset_names = all_dataset_names[len(all_dataset_names) // 2 :]
    return get_combined_train_dataset(
        all_dataset_names=all_dataset_names,
        *args,
        **kwargs,
        max_n=800_000,
    )
