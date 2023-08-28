import os
import string
import torch

from .registry import register_dataset
from .llm_utils import tokenize_for_causal_lm


__all__ = [
    "get_story_cloze",
]


def __format_prompt(sample, style, with_answer=False):
    if style == "choice":
        story = " ".join([sample[f"input_sentence_{i}"] for i in range(1, 5)])
        answer_map = [sample["sentence_quiz1"], sample["sentence_quiz2"]]
        answer = string.ascii_lowercase[sample["answer_right_ending"] - 1] + "</s>\n"

        prompt = "\n".join(
            [
                "Story:",
                story,
                "\nChoices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(
                        string.ascii_lowercase[: len(answer_map)], answer_map
                    )
                ],
                f"Answer: {answer if with_answer else ''}",
            ]
        )

        return prompt

    raise NotImplementedError


def __generate_fewshot_prompts(dataset, prompt_style, kshot, seed=None):
    if kshot <= 0:
        return ""

    fewshot_prompt = "\n".join(
        [
            "The following are stories (with completitions).\n",
            *[
                __format_prompt(dataset[idx], prompt_style, with_answer=True)
                for idx in torch.randperm(
                    len(dataset), generator=torch.Generator().manual_seed(seed)
                )[:kshot].tolist()
            ],
        ]
    )
    fewshot_prompt = fewshot_prompt + "\nNow, answer the next question.\n\n"

    return fewshot_prompt


def get_story_cloze(
    root=None,
    prompt_style=None,
    eval_kshot=0,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    from datasets import load_dataset

    ## NOTE: needs manual download.
    dataset = load_dataset(
        "story_cloze",
        "2018",
        data_dir=f"{os.environ.get('HF_DATASETS_CACHE', root)}/story_cloze",
        cache_dir=os.environ.get("HF_DATASETS_CACHE", root),
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    (val_data,) = [
        data.map(
            lambda x: {
                "source": __generate_fewshot_prompts(data, prompt_style, k, seed=seed)
                + __format_prompt(x, prompt_style),
                "target": f"{string.ascii_lowercase[x['answer_right_ending'] - 1]}{tokenizer.eos_token}",
            },
            num_proc=num_workers,
            remove_columns=[
                "answer_right_ending",
                "input_sentence_1",
                "input_sentence_2",
                "input_sentence_3",
                "input_sentence_4",
                "sentence_quiz1",
                "sentence_quiz2",
                "story_id",
            ],
        ).map(
            lambda x: tokenize_for_causal_lm(tokenizer, x),
            num_proc=num_workers,
            remove_columns=["source", "target"],
        )
        for data, k in zip([dataset.pop("validation")], [eval_kshot])
    ]

    return None, val_data, None


@register_dataset(attrs=dict(task_tags=["commonsense"]))
def story_cloze(*args, **kwargs):
    return get_story_cloze(
        *args,
        **kwargs,
        prompt_style="choice",
    )
