import os
import glob

from .registry import register_dataset
from .llm_utils import LMText
from .data_collator import LabeledStringDataCollator


def get_offline(
    root=None,
    num_workers=8,
    use_cache=True,
    tokenizer=None,
    max_token_length=None,
    train_kshot=0,
    eval_kshot=0,
    **_,
):
    from datasets import load_dataset, Features, Value

    features = Features(
        {
            "context": Value("string"),
            "target": Value("string"),
            "target_prompt": Value("string"),
            "prompt": Value("string"),
            # "source_dataset": Value("string"),
            "output": Value("string"),
            "query_label": Value("int32"),
        }
    )

    data_files = {}
    for split_name in ["train", "validation", "test"]:
        if os.path.isdir(f"{root}/{split_name}"):
            data_files[split_name] = glob.glob(f"{root}/{split_name}/*.csv")

    dataset = load_dataset(
        "csv",
        data_files=data_files,
        cache_dir=os.environ.get("HF_DATASETS_CACHE", root),
        features=features,
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    def _replace_none(x):
        return {k: "" if v is None else v for k, v in x.items()}

    data_splits = {
        split: dataset[split].map(
            _replace_none,
            num_proc=num_workers,
        )
        for split in data_files.keys()
    }

    data_kshot = {
        "train": train_kshot,
        "validation": eval_kshot,
        "test": eval_kshot,
    }

    data_splits = {
        k: (
            ds.remove_columns([c for c in ["prompt"] if c in ds.column_names])
            if data_kshot[k] == 0
            else ds
        )
        for k, ds in data_splits.items()
    }

    if max_token_length is not None:
        tokenizer_args = LabeledStringDataCollator.get_tokenizer_args(tokenizer)

        def token_length_filter(instance):
            inputs = tokenizer(
                [str(LMText.from_(instance))],
                **tokenizer_args,
            )
            return inputs.get("input_ids").size(-1) <= max_token_length

        data_splits = {
            k: ds.filter(token_length_filter, num_proc=num_workers)
            for k, ds in data_splits.items()
        }

    return (
        data_splits.pop("train", None),
        data_splits.pop("validation", None),
        data_splits.pop("test", None),
    )


@register_dataset
def offline(*args, root=None, dataset_str=None, max_token_length=None, **kwargs):
    if len(dataset_str.split(":")) == 2:
        root = dataset_str.split(":")[1]

    return get_offline(
        *args,
        **kwargs,
        root=root,
        max_token_length=max_token_length,
    )


@register_dataset
def offline_noprompt(*args, **kwargs):
    kwargs.pop("train_kshot", None)
    kwargs.pop("eval_kshot", None)
    return offline(*args, **kwargs, train_kshot=0, eval_kshot=0)
