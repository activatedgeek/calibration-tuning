import os
import string
import torch

from .registry import register_dataset
from .llm_utils import tokenize_for_causal_lm


__all__ = [
    "get_wsc",
]


def __format_sample(sample, tokenizer, style):
    if style == "choice":
        text = sample["text"]
        answer_map = sample["options"]

        source = "\n".join(
            [
                "Sentence:",
                text,
                "\nChoices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(
                        string.ascii_lowercase[: len(answer_map)], answer_map
                    )
                ],
                f"Answer: ",
            ]
        )

        target = string.ascii_lowercase[sample["label"]] + tokenizer.eos_token

        return dict(source=source, target=target)

    raise NotImplementedError


def __generate_fewshot_prompts(
    tokenizer, prompt_style, prompt_dataset, kshot, seed=None
):
    if kshot <= 0:
        return ""

    _c = lambda s: s["source"] + s["target"]

    fewshot_prompt = "\n".join(
        [
            "The following are sentences with ambiguity.\n",
            *[
                _c(__format_sample(prompt_dataset[idx], tokenizer, prompt_style)) + "\n"
                for idx in torch.randperm(
                    len(prompt_dataset), generator=torch.Generator().manual_seed(seed)
                )[:kshot].tolist()
            ],
        ]
    )
    fewshot_prompt = fewshot_prompt + "\nNow, resolve the next ambiguity."

    return fewshot_prompt


def __format_sample_with_prompt(
    sample, tokenizer, prompt_style, prompt_dataset, kshot, seed=None
):
    sample_dict = __format_sample(sample, tokenizer, prompt_style)

    prompt_str = __generate_fewshot_prompts(
        tokenizer, prompt_style, prompt_dataset, kshot, seed=seed
    )
    if len(prompt_str):
        prompt_str += "\n\n"

    sample_dict["source"] = prompt_str + sample_dict["source"]

    return sample_dict


def get_wsc(
    root=None,
    subset=None,
    prompt_style=None,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "winograd_wsc", subset, cache_dir=os.environ.get("HF_DATASETS_CACHE", root)
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    test_data = dataset.pop("test")
    test_data = test_data.map(
        lambda x: __format_sample_with_prompt(
            x, tokenizer, prompt_style, test_data, eval_kshot, seed=seed
        ),
        num_proc=num_workers,
        remove_columns=[
            "text",
            "pronoun",
            "pronoun_loc",
            "quote",
            "quote_loc",
            "options",
            "label",
            "source",
        ],
    ).map(
        lambda x: tokenize_for_causal_lm(tokenizer, x),
        num_proc=num_workers,
        remove_columns=["source", "target"],
    )

    return None, None, test_data


@register_dataset(attrs=dict(task_tags=["coreference"]))
def wsc(*args, prompt_style="choice", **kwargs):
    return get_wsc(
        *args,
        **kwargs,
        subset="wsc285",
        prompt_style=prompt_style,
    )
