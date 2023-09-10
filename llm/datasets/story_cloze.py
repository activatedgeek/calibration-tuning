import os
import string
import torch

from .registry import register_dataset


__all__ = [
    "get_story_cloze",
]


def __format_sample(sample, tokenizer, style):
    if style == "choice":
        story = " ".join([sample[f"input_sentence_{i}"] for i in range(1, 5)])
        answer_map = [sample["sentence_quiz1"], sample["sentence_quiz2"]]

        source = "\n".join(
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
                f"Answer: ",
            ]
        )

        target = (
            string.ascii_lowercase[sample["answer_right_ending"] - 1]
            + tokenizer.eos_token
        )

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
            "The following are stories (with completions).\n",
            *[
                _c(__format_sample(prompt_dataset[idx], tokenizer, prompt_style)) + "\n"
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
    sample_dict = __format_sample(sample, tokenizer, prompt_style)

    prompt_str = __generate_fewshot_prompts(
        tokenizer, prompt_style, prompt_dataset, kshot, seed=seed
    )
    if len(prompt_str):
        prompt_str += "\n\n"

    sample_dict["source"] = prompt_str + sample_dict["source"]

    return sample_dict


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
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, k, seed=seed
            ),
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
        )
        for data, k in zip([dataset.pop("validation")], [eval_kshot])
    ]

    return None, val_data, None


@register_dataset(attrs=dict(task_tags=["commonsense"]))
def story_cloze(*args, prompt_style="choice", **kwargs):
    return get_story_cloze(
        *args,
        **kwargs,
        prompt_style=prompt_style,
    )
