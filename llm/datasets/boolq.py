import string
import torch
from datasets import load_dataset

from .registry import register_dataset
from .llm_utils import LMText


__all__ = [
    "get_boolq",
]


def __format_sample(sample, tokenizer, style):
    target_prompt = "\nAnswer:"

    passage = sample["passage"]
    question = sample["question"]
    answer_map = ["False", "True"]

    if style == "choice":
        context = "\n".join(
            [
                "Passage:",
                passage,
                "\nQuestion:",
                question,
                "\nChoices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(
                        string.ascii_lowercase[: len(answer_map)], answer_map
                    )
                ],
            ]
        )

        target = string.ascii_lowercase[int(bool(sample["answer"]))]
    elif style == "oe":
        context = "\n".join(
            [
                "Read the following passage and answer if the question is true or false.",
                f"Passage: {passage}\n",
                f"Question: {question}?\n",
                'Is the answer True or False? Respond with only "True" or "False" and no additional text.',
            ]
        )

        target = str(sample["answer"])
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
            "The following are comprehension passages with answers.\n",
            *[
                str(__format_sample(prompt_dataset[idx], tokenizer, prompt_style))
                + "\n"
                for idx in torch.randperm(
                    len(prompt_dataset), generator=torch.Generator().manual_seed(seed)
                )[:kshot].tolist()
            ],
        ]
    )
    fewshot_prompt = (
        fewshot_prompt + "\nNow, answer the next question after the passage."
    )

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


def get_boolq(
    prompt_style=None,
    train_kshot=0,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    dataset = load_dataset("boolq")
    if not use_cache:
        dataset.cleanup_cache_files()

    train_data, val_data = [
        data.map(
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, k, seed=seed
            ).to_pydict(),
            num_proc=num_workers,
            remove_columns=[
                "passage",
                "question",
                "answer",
            ],
        )
        for data, k in zip(
            [dataset.pop("train"), dataset.pop("validation")],
            [train_kshot, eval_kshot],
        )
    ]

    return train_data, val_data, None


@register_dataset(attrs=dict(task_tags=["comprehension"]))
def boolq(*args, prompt_style="choice", **kwargs):
    return get_boolq(
        *args,
        **kwargs,
        prompt_style=prompt_style,
    )
