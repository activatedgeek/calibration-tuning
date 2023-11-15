import os
import string
import torch
import numpy as np

from .registry import register_dataset
from .llm_utils import LMText


__all__ = [
    "get_bigbench_mc",
]

__MC_TASKS = [
    "abstract_narrative_understanding",
    "anachronisms",
    "analogical_similarity",
    "analytic_entailment",
    "arithmetic",
    "authorship_verification",
    "bbq_lite_json",
    "causal_judgment",
    "cause_and_effect",
    "checkmate_in_one",
    "code_line_description",
    "color",
    "common_morpheme",
    "conceptual_combinations",
    "contextual_parametric_knowledge_conflicts",
    "crash_blossom",
    "crass_ai",
    "cryobiology_spanish",
    "cs_algorithms",
    "dark_humor_detection",
    "date_understanding",
    "disambiguation_qa",
    "discourse_marker_prediction",
    "dyck_languages",
    "elementary_math_qa",
    "emoji_movie",
    "emojis_emotion_prediction",
    "empirical_judgments",
    "english_proverbs",
    "english_russian_proverbs",
    "entailed_polarity",
    "entailed_polarity_hindi",
    "epistemic_reasoning",
    "evaluating_information_essentiality",
    "fact_checker",
    "fantasy_reasoning",
    "figure_of_speech_detection",
    "formal_fallacies_syllogisms_negation",
    "general_knowledge",
    "geometric_shapes",
    "goal_step_wikihow",
    "gre_reading_comprehension",
    "hhh_alignment",
    "hindu_knowledge",
    "hinglish_toxicity",
    "human_organs_senses",
    "hyperbaton",
    "identify_math_theorems",
    "identify_odd_metaphor",
    "implicatures",
    "implicit_relations",
    "indic_cause_and_effect",
    "intent_recognition",
    "international_phonetic_alphabet_nli",
    "intersect_geometry",
    "irony_identification",
    # "kanji_ascii",
    "kannada",
    "key_value_maps",
    "known_unknowns",
    "language_identification",
    "logic_grid_puzzle",
    "logical_args",
    "logical_deduction",
    "logical_fallacy_detection",
    "logical_sequence",
    "mathematical_induction",
    "medical_questions_russian",
    "metaphor_boolean",
    "metaphor_understanding",
    "misconceptions",
    "misconceptions_russian",
    "mnist_ascii",
    "moral_permissibility",
    "movie_dialog_same_or_different",
    "movie_recommendation",
    "navigate",
    "nonsense_words_grammar",
    "novel_concepts",
    "odd_one_out",
    "parsinlu_qa",
    "penguins_in_a_table",
    "periodic_elements",
    "persian_idioms",
    "phrase_relatedness",
    "physical_intuition",
    "physics",
    "play_dialog_same_or_different",
    "presuppositions_as_nli",
    "question_selection",
    "real_or_fake_text",
    "reasoning_about_colored_objects",
    "rhyming",
    "riddle_sense",
    "ruin_names",
    "salient_translation_error_detection",
    "sentence_ambiguity",
    "similarities_abstraction",
    "simple_arithmetic_json_multiple_choice",
    "simple_ethical_questions",
    "snarks",
    "social_iqa",
    "social_support",
    "sports_understanding",
    "strange_stories",
    "strategyqa",
    "suicide_risk",
    "swahili_english_proverbs",
    "swedish_to_german_proverbs",
    "symbol_interpretation",
    "temporal_sequences",
    "timedial",
    "tracking_shuffled_objects",
    "understanding_fables",
    "undo_permutation",
    "unit_conversion",
    "unit_interpretation",
    "vitaminc_fact_verification",
    "what_is_the_tao",
    "winowhy",
]


def __format_sample(sample, tokenizer, style):
    target_prompt = "\nAnswer: "

    if style == "choice":
        question = sample["inputs"]
        choices = sample["multiple_choice_targets"]

        context = "\n".join(
            [
                f"Question:\n{question}",
                "\nChoices:",
                *[
                    f"  ({n}): {c}"
                    for n, c in zip(string.ascii_lowercase[: len(choices)], choices)
                ],
            ]
        )

        target = (
            string.ascii_lowercase[np.argmax(sample["multiple_choice_scores"])]
            + tokenizer.eos_token
        )
    else:
        raise NotImplementedError

    return LMText(context=context, target_prompt=target_prompt, target=target)


def __generate_fewshot_prompts(
    tokenizer, prompt_style, prompt_dataset, subset, kshot, seed=None
):
    if kshot <= 0:
        return ""

    fewshot_prompt = "\n".join(
        [
            f"The following are multiple choice questions (with answers) about {' '.join(subset.split('_'))}.\n",
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
    sample, tokenizer, prompt_style, prompt_dataset, instance, kshot, seed=None
):
    prompt = __generate_fewshot_prompts(
        tokenizer, prompt_style, prompt_dataset, instance, kshot, seed=seed
    )
    if len(prompt):
        prompt += "\n\n"

    sample = __format_sample(sample, tokenizer, prompt_style)
    sample.prompt = prompt

    return sample


def get_bigbench_mc(
    root=None,
    subset=None,
    prompt_style=None,
    eval_kshot=1,
    tokenizer=None,
    num_workers=8,
    seed=None,
    use_cache=True,
    **_,
):
    from datasets import load_dataset

    dataset = load_dataset(
        "tasksource/bigbench",
        subset,
        cache_dir=os.environ.get("HF_DATASETS_CACHE", root),
    )
    if not use_cache:
        dataset.cleanup_cache_files()

    train_data, val_data = [
        data.filter(
            lambda x: bool(
                len(x["multiple_choice_scores"])
                and (len(x["multiple_choice_scores"]) <= len(string.ascii_lowercase))
            ),
            num_proc=num_workers,
        ).map(
            lambda x: __format_sample_with_prompt(
                x, tokenizer, prompt_style, data, subset, k, seed=seed
            ).to_pydict(),
            num_proc=num_workers,
            remove_columns=[
                "inputs",
                "targets",
                "multiple_choice_targets",
                "multiple_choice_scores",
                "idx",
            ],
        )
        for data, k in zip(
            [dataset.pop("train"), dataset.pop("validation")],
            [0, eval_kshot],
        )
    ]

    return train_data, val_data, None


@register_dataset(attrs=dict(tasks=__MC_TASKS))
def bbmc(*args, dataset_str=None, prompt_style="choice", **kwargs):
    d, subset = dataset_str.split(":")

    assert d == "bbmc" and isinstance(
        subset, str
    ), f"Dataset string should be formatted as 'bbmc:<subset>', found {dataset_str}"

    assert subset in __MC_TASKS, f"Task '{subset}' not found."

    return get_bigbench_mc(
        *args,
        **kwargs,
        subset=subset,
        prompt_style=prompt_style,
    )
