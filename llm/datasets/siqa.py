import os
import string
import torch

from .registry import register_dataset
from .llm_utils import LMText


__all__ = [
    "get_sciq",
]


def __format_sample(sample, tokenizer, style):
    target_prompt = "\nAnswer:"

    context = sample["context"]
    question = sample["question"]

    answer_map = [
        sample["answerA"],
        sample["answerB"],
        sample["answerC"],
    ]

    if style == "choice":
        context = "\n".join(
            [
                f"Context:\n{context}",
                f"\nQuestion:\n{question}",
                "\nChoices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(
                        string.ascii_lowercase[: len(answer_map)], answer_map
                    )
                ],
            ]
        )

        target = string.ascii_lowercase[int(sample["label"]) - 1]
    elif style == "oe":
        context = "\n".join(
            [
                "Read the following paragraph and answer the question. Give ONLY the answer, no other words or explanation.\n",
                "For example:\n",
                "Answer: <most likely answer, as short as possible>.\n",
                f"Paragraph: {context}\n",
                f"Question: {question}",
            ]
        )

        target = answer_map[int(sample["label"]) - 1]
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
            "The following are multiple choice social interaction questions.\n",
            *[
                str(__format_sample(prompt_dataset[idx], tokenizer, prompt_style))
                + "\n"
                for idx in torch.randperm(
                    len(prompt_dataset), generator=torch.Generator().manual_seed(seed)
                )[:kshot].tolist()
            ],
        ]
    )
    fewshot_prompt = fewshot_prompt + "\nNow, answer the next question."

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


def get_siqa(
    root=None,
    prompt_style=None,
    train_kshot=0,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "social_i_qa",
        cache_dir=os.environ.get("HF_DATASETS_CACHE", root),
        trust_remote_code=True,
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    train_data, val_data = [
        data.map(
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, k, seed=seed
            ).to_pydict(),
            num_proc=num_workers,
            remove_columns=[
                "context",
                "question",
                "answerA",
                "answerB",
                "answerC",
                "label",
            ],
        )
        for data, k in zip(
            [dataset.pop("train"), dataset.pop("validation")],
            [train_kshot, eval_kshot],
        )
    ]

    return train_data, val_data, None


@register_dataset(attrs=dict(task_tags=["reasoning"]))
def siqa(*args, prompt_style="choice", **kwargs):
    return get_siqa(
        *args,
        **kwargs,
        prompt_style=prompt_style,
    )
