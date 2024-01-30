import numpy as np
from datasets import concatenate_datasets

from .registry import get_dataset, list_datasets, register_dataset, get_dataset_attrs
from ..random import FixedSeed


def get_all_datasets_list(dataset_str, prompt_style=None):
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
                            s in x for s in ["all", "sub", "mmlu", "bbmc", "offline"]
                        ),
                        list_datasets(),
                    )
                )
            )
            ## Skip datasets for oe.
            if prompt_style == "oe":
                all_datasets_list = list(
                    filter(
                        lambda x: not any(s in x for s in ["hellaswag"]),
                        all_datasets_list,
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


def _concat_datasets(datasets, max_n, complement=False, uniform=False):
    all_n = [len(ds) for ds in datasets]
    total_n = min(max_n, sum(all_n))

    if uniform:
        equal_n = max_n // len(all_n)
        select_n = [min(equal_n, len(ds)) for ds in datasets]

        if complement:
            return concatenate_datasets(
                [
                    ds.select(range(n, N))
                    for ds, N, n in zip(datasets, all_n, select_n)
                    if n < N
                ]
            )

        return concatenate_datasets(
            [ds.select(range(n)) for ds, n in zip(datasets, select_n)]
        )

    select_n = ((np.array(all_n) / sum(all_n)) * total_n).astype(int)

    return concatenate_datasets(
        [
            ds.select(range(n, N) if complement else range(n))
            for ds, N, n in zip(datasets, all_n, select_n)
        ]
    )


def get_combined_dataset(
    all_dataset_names,
    max_n=100,
    seed=None,
    complement=False,
    uniform=False,
    **kwargs,
):
    all_train_data, all_val_data, all_test_data = [], [], []
    for dataset in all_dataset_names:
        train_data, val_data, test_data = get_dataset(
            dataset,
            seed=seed,
            **kwargs,
        )

        if train_data is not None:
            train_data = train_data.shuffle(seed=seed)
            if "source_dataset" in train_data.column_names:
                train_data = train_data.remove_columns(["source_dataset"])
            train_data = train_data.add_column(
                "source_dataset", [dataset] * len(train_data)
            )
            all_train_data.append(train_data)

        if val_data is not None:
            val_data = val_data.shuffle(seed=seed)
            all_val_data.append(val_data)

        if test_data is not None:
            test_data = test_data.shuffle(seed=seed)
            all_test_data.append(test_data)

    all_train_data = _concat_datasets(
        all_train_data, max_n, complement=complement, uniform=uniform
    )
    all_val_data = _concat_datasets(all_val_data, max_n)
    all_test_data = _concat_datasets(all_test_data, max_n)

    return all_train_data, all_val_data, all_test_data


@register_dataset
def all_200k(
    *args, max_n=200_000, max_val_n=None, prompt_style="choice", seed=137, complement=False, **kwargs
):
    tr, vl, _ = get_combined_dataset(
        all_dataset_names=get_all_datasets_list("all:train", prompt_style=prompt_style),
        *args,
        **kwargs,
        seed=seed,
        prompt_style=prompt_style,
        max_n=max_n,
        complement=complement,
    )

    with FixedSeed(seed):
        max_val_n = max_val_n or max_n
        vl = vl.select(
            np.random.choice(
                range(min(len(vl), max_val_n)), min(len(vl), max_val_n), replace=False
            )
        )

    return tr, vl, None


@register_dataset
def all_20k_uniform(*args, max_n=20_000, max_val_n=5_000, **kwargs):
    return all_200k(*args, max_n=max_n, max_val_n=max_val_n, uniform=True, **kwargs)


@register_dataset
def all_100_uniform(*args, max_n=100, **kwargs):
    return all_20k_uniform(*args, max_n=max_n, **kwargs)


@register_dataset
def cal_all_50k(*args, max_n=50_000, prompt_style="choice", **kwargs):
    _, vl, _ = get_combined_dataset(
        all_dataset_names=get_all_datasets_list("all:train", prompt_style=prompt_style),
        *args,
        **kwargs,
        prompt_style=prompt_style,
        eval_kshot=0,
        max_n=max_n,
        complement=False,
    )
    return vl, None, None


@register_dataset
def all_200k_c(*args, max_n=200_000, prompt_style="choice", **kwargs):
    tr, _, _ = get_combined_dataset(
        all_dataset_names=get_all_datasets_list("all:train", prompt_style=prompt_style),
        *args,
        **kwargs,
        prompt_style=prompt_style,
        max_n=max_n,
        complement=True,
    )
    return tr, None, None


@register_dataset
def sub_200k(*args, max_n=200_000, prompt_style="choice", **kwargs):
    all_dataset_names = get_all_datasets_list("all:train", prompt_style=prompt_style)
    all_dataset_names = all_dataset_names[: len(all_dataset_names) // 2]
    tr, _, _ = get_combined_dataset(
        all_dataset_names=all_dataset_names,
        *args,
        **kwargs,
        prompt_style=prompt_style,
        max_n=max_n,
        complement=False,
    )
    return tr, None, None


@register_dataset
def cal_sub_200k(*args, max_n=200_000, prompt_style="choice", **kwargs):
    all_dataset_names = get_all_datasets_list("all:train", prompt_style=prompt_style)
    all_dataset_names = all_dataset_names[: len(all_dataset_names) // 2]
    _, vl, _ = get_combined_dataset(
        all_dataset_names=all_dataset_names,
        *args,
        **kwargs,
        prompt_style=prompt_style,
        max_n=max_n,
        complement=False,
    )
    return vl, None, None


@register_dataset
def sub_200k_c(*args, max_n=800_000, prompt_style="choice", **kwargs):
    all_dataset_names = get_all_datasets_list("all:train", prompt_style=prompt_style)
    all_dataset_names = all_dataset_names[len(all_dataset_names) // 2 :]

    # all_dataset_names = all_dataset_names[3:]
    # print(all_dataset_names)
    # print(1/0)

    tr, _, _ = get_combined_dataset(
        all_dataset_names=all_dataset_names,
        *args,
        **kwargs,
        prompt_style=prompt_style,
        max_n=max_n,
    )
    return tr, None, None


@register_dataset
def cal_sub_200k_c(*args, max_n=800_000, prompt_style="choice", **kwargs):
    all_dataset_names = get_all_datasets_list("all:train", prompt_style=prompt_style)
    all_dataset_names = all_dataset_names[len(all_dataset_names) // 2 :]
    _, vl, _ = get_combined_dataset(
        all_dataset_names=all_dataset_names,
        *args,
        **kwargs,
        prompt_style=prompt_style,
        max_n=max_n,
    )
    return vl, None, None


@register_dataset
def cal_mmlu(*args, max_n=50_000, **kwargs):
    mmlu_datasets = [f"mmlu:{task}" for task in get_dataset_attrs("mmlu").get("tasks")]

    tr, _, _ = get_combined_dataset(
        all_dataset_names=mmlu_datasets,
        *args,
        **kwargs,
        max_n=max_n,
    )
    return tr, None, None
