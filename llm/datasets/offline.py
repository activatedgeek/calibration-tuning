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

    train_data, val_data = [
        dataset[split].map(
            _replace_none,
            num_proc=num_workers,
            remove_columns=["prompt"] if k == 0 else [],
        )
        for split, k in zip(data_files.keys(), [train_kshot, eval_kshot])
    ]

    if max_token_length is not None:
        tokenizer_args = LabeledStringDataCollator.get_tokenizer_args(tokenizer)

        def token_length_filter(instance):
            inputs = tokenizer(
                [str(LMText.from_(instance))],
                **tokenizer_args,
            )
            return inputs.get("input_ids").size(-1) <= max_token_length

        train_data = train_data.filter(token_length_filter, num_proc=num_workers)
        val_data = val_data.filter(token_length_filter, num_proc=num_workers)

    return train_data, val_data, None


@register_dataset
def offline(*args, max_token_length=None, **kwargs):
    return get_offline(
        *args,
        **kwargs,
        max_token_length=max_token_length,
    )
