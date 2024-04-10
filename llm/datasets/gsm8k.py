import torch
from datasets import load_dataset

from .registry import register_dataset
from .llm_utils import LMText


__all__ = [
    "get_gsm8k",
]


def __format_sample(sample, tokenizer, style):
    context = sample["question"]

    if style == "oe":
        context = "\n".join(["Question:", context])
        target_prompt = "\nAnswer:"
        target = sample["answer"]
    else:
        raise NotImplementedError

    return LMText(context=context, target_prompt=target_prompt, target=target)


def __generate_fewshot_prompts(
    tokenizer, prompt_style, prompt_dataset, kshot, seed=None
):
    if kshot <= 0:
        return ""

    fewshot_prompt = "\n".join(
        [
            "The following are math questions (with answers).\n",
            *[
                str(__format_sample(prompt_dataset[idx], tokenizer, prompt_style))
                + "\n"
                for idx in torch.randperm(
                    len(prompt_dataset), generator=torch.Generator().manual_seed(seed)
                )[:kshot].tolist()
            ],
        ]
    )
    fewshot_prompt = fewshot_prompt + "\nNow, answer the following."

    return fewshot_prompt


def __format_sample_with_prompt(
    sample, tokenizer, prompt_style, prompt_dataset, kshot, seed=None
):
    prompt = __generate_fewshot_prompts(
        tokenizer, prompt_style, prompt_dataset, kshot, seed=seed
    )
    if len(prompt):
        prompt += "\n\n"

    sample = __format_sample(sample, tokenizer, prompt_style)
    sample.prompt = prompt

    return sample


def get_gsm8k(
    prompt_style=None,
    train_kshot=0,
    eval_kshot=8,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    dataset = load_dataset("gsm8k", "main")
    if not use_cache:
        dataset.cleanup_cache_files()

    train_data, test_data = [
        data.map(
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, k, seed=seed
            ).to_pydict(),
            num_proc=num_workers,
            remove_columns=["question", "answer"],
        )
        for data, k in zip(
            [dataset.pop("train"), dataset.pop("test")],
            [train_kshot, eval_kshot],
        )
    ]

    return train_data, None, test_data


@register_dataset
def gsm8k(*args, prompt_style="oe", **kwargs):
    return get_gsm8k(
        *args,
        **kwargs,
        prompt_style=prompt_style,
    )
