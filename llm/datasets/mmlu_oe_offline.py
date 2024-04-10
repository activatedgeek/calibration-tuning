import os
import glob
from datasets import load_dataset, Features, Value

from .registry import register_dataset


def get_mmlu_oe_offline(
    root=None,
    num_workers=8,
    use_cache=True,
    **_,
):
    features = Features(
        {
            "context": Value("string"),
            "target": Value("string"),
            "target_prompt": Value("string"),
            "prompt": Value("string"),
            "output": Value("string"),
            "query_label": Value("int32"),
        }
    )

    data_files = {}
    for split_name in ["validation", "test"]:
        if os.path.isdir(f"{root}/{split_name}"):
            data_files[split_name] = glob.glob(f"{root}/{split_name}/*.csv")

    dataset = load_dataset("csv", data_files=data_files, features=features)
    if not use_cache:
        dataset.cleanup_cache_files()

    def _check(sample):
        if sample["output"] is None:
            sample["output"] = ""
        return sample

    dataset = dataset.map(_check, num_proc=num_workers)

    val_data = dataset.pop("validation")
    test_data = dataset.pop("test")

    return None, val_data, test_data


@register_dataset
def mmlu_oe_offline(*args, root=None, dataset_str=None, **kwargs):
    if len(dataset_str.split(":")) == 2:
        root = dataset_str.split(":")[1]

    return get_mmlu_oe_offline(
        *args,
        **kwargs,
        root=root,
    )
