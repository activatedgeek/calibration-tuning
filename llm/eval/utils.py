import logging
from functools import partial

from ..logging import Timer
from ..datasets import get_dataset, get_loader
from .choice import (
    evaluate_contextual_calibration_choice,
    evaluate_candidate_choice,
    evaluate_choice,
    evaluate_classifier_choice,
)
from .oe import (
    evaluate_oe,
    evaluate_classifier_oe,
    evaluate_uncertainty_sampling_oe,
    evaluate_verbal_elicitation_oe,
)

EVALUATE_MODE_FN_MAP = {
    "choice": evaluate_choice,
    "cc_choice": evaluate_contextual_calibration_choice,
    "cand_choice": evaluate_candidate_choice,
    "class_choice": evaluate_classifier_choice,
    "oe": evaluate_oe,
    "us_oe": evaluate_uncertainty_sampling_oe,
    "class_oe": evaluate_classifier_oe,
    "ve_oe": evaluate_verbal_elicitation_oe,
}

VERBAL_ELICITATION_MAP = {
    "1s1g": {
        "prompt": "".join(
            [
                "Provide your best guess and the probability that it is correct (0.0 to 1.0) for ",
                "the following question. Give ONLY the guess and probability, no other words or ",
                "explanation. For example:\n\nGuess: <most likely guess, as short as possible; not ",
                "a complete sentence, just the guess!>\nProbability: <the probability between 0.0 ",
                "and 1.0 that your guess is correct, without any extra commentary whatsoever; just ",
                "the probability!>\n\n",
            ]
        ),
        "target_prompt": "\nGuess: ",
    },
    "1s2g": {
        "prompt": "".join(
            [
                "Provide your 2 best guesses and the probability that each is correct (0.0 to ",
                "1.0) for the following question. Give ONLY the guesses and probabilities, no other ",
                "words or explanation. For example:\n\nG1: <first most likely guess, as short as ",
                "possible; not a complete sentence, just the guess!>\n\nP1: <the probability between ",
                "0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the ",
                "probability!>\n\nG2: <second most likely guess, as short as possible; ",
                "not a complete sentence, just the guess!>\n\nP2: <the probability between 0.0 ",
                "and 1.0 that G2 is correct, without any extra commentary whatsoever; just the ",
                "probability!>\n\n",
            ]
        ),
        "target_prompt": "\nG1: ",
    },
    "1s4g": {
        "prompt": "".join(
            [
                "Provide your 4 best guesses and the probability that each is correct (0.0 to ",
                "1.0) for the following question. Give ONLY the guesses and probabilities, no other ",
                "words or explanation. For example:\n\nG1: <first most likely guess, as short as ",
                "possible; not a complete sentence, just the guess!>\n\nP1: <the probability between ",
                "0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the ",
                "probability!>\n\nG2: <second most likely guess, as short as possible; ",
                "not a complete sentence, just the guess!>\n\nP2: <the probability between 0.0 ",
                "and 1.0 that G2 is correct, without any extra commentary whatsoever; just the ",
                "probability!>\n\nG3: <third most likely guess, as short as possible; ",
                "not a complete sentence, just the guess!>\n\nP3: <the probability between 0.0 ",
                "and 1.0 that G3 is correct, without any extra commentary whatsoever; just the ",
                "probability!>\n\nG4: <fourth most likely guess, as short as possible; ",
                "not a complete sentence, just the guess!>\n\nP4: <the probability between 0.0 ",
                "and 1.0 that G4 is correct, without any extra commentary whatsoever; just the ",
                "probability!>\n\n",
            ]
        ),
        "target_prompt": "\nG1: ",
    },
    "2s1CoT": {
        "prompt": "".join(
            [
                "Provide your best guess for the following question. Before giving your answer, ",
                "provide a step-by-step explanation of your thought process. Then on a new line ",
                "give the guess with no other words or explanation.\n\nFor example:\n\nExplanation: ",
                "<one sentence step-by-step explanation of your thought process>\n\nGuess: ",
                "<most likely guess, as short as possible; not a complete sentence, just the ",
                "guess!>.\n\n",
            ]
        ),
        "target_prompt": "\nExplanation: ",
    },
    "2s1g": {
        "prompt": "".join(
            [
                "Provide your best guess for the following question. Give ONLY the guess, no ",
                "other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, as ",
                "short as possible; not a complete sentence, just the guess!>\n\n",
            ]
        ),
        "target_prompt": "\nGuess: ",
    },
    "2s2g": {
        "prompt": "".join(
            [
                "Provide your 2 best guesses for the following question. Give ONLY the guesses, ",
                "no other words or explanation. For example:\n\nG1: <first most likely guess, as ",
                "short as possible; not a complete sentence, just the guess!>\n\nP1: <the probability ",
                "between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; ",
                "just the probability!>\n\nG2: <second most likely guess, as short as possible; ",
                "not a complete sentence, just the guess!>\n\n",
            ]
        ),
        "target_prompt": "\nG1: ",
    },
    "2s4g": {
        "prompt": "".join(
            [
                "Provide your 4 best guesses for the following question. Give ONLY the guesses, ",
                "no other words or explanation. For example:\n\nG1: <first most likely guess, as ",
                "short as possible; not a complete sentence, just the guess!>\n\nP1: <the probability ",
                "between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; ",
                "just the probability!>\n\nG2: <second most likely guess, as short as possible; ",
                "not a complete sentence, just the guess!>\n\nP2: <the probability between 0.0 ",
                "and 1.0 that G2 is correct, without any extra commentary whatsoever; just the ",
                "probability!>\n\nG3: <third most likely guess, as short as possible; ",
                "not a complete sentence, just the guess!>\n\nG4: <fourth most likely guess, ",
                "as short as possible; not a complete sentence, just the guess!>\n\n",
            ]
        ),
        "target_prompt": "\nG1: ",
    },
}


def evaluate_dataset(
    accelerator,
    model,
    tokenizer,
    dataset,
    data_dir=None,
    train_data=None,
    val_data=None,
    test_data=None,
    seed=137,
    batch_size=1,
    num_workers=8,
    eval_kshot=None,
    use_cache=True,
    prompt_style=None,
    log_dir=None,
    evaluate_fn=None,
):
    if dataset is not None:
        with accelerator.main_process_first():
            _extra_args = dict()
            ## NOTE: Conditional to avoid overriding default kshot specification in dataset definition.
            if eval_kshot is not None:
                _extra_args["eval_kshot"] = eval_kshot
            data_splits = get_dataset(
                dataset,
                root=data_dir,
                tokenizer=tokenizer,
                seed=seed,
                num_workers=num_workers,
                use_cache=use_cache,
                prompt_style=prompt_style,
                **_extra_args,
            )
    else:
        if (val_data is not None) and (test_data is not None):
            logging.warning(f"Missing val_data or test_data.")

        data_splits = (train_data, val_data, test_data)

    if train_data == False:
        data_splits = (None, *data_splits[1:])
    data_splits = [
        (s, ds)
        for s, ds in zip(["train", "validation", "test"], data_splits)
        if ds is not None
    ]

    if "ve" in evaluate_fn:
        style = evaluate_fn.split("_")[1]

        def switch_prompt(example):
            example["prompt"] = VERBAL_ELICITATION_MAP[style]["prompt"]
            example["target_prompt"] = VERBAL_ELICITATION_MAP[style]["target_prompt"]
            return example

        if train_data:
            train_data = train_data.map(switch_prompt)
        val_data = val_data.map(switch_prompt)
        test_data = test_data.map(switch_prompt)

    if isinstance(evaluate_fn, str):
        if "cc_oe_" in evaluate_fn:
            assert evaluate_fn[:6] == "cc_oe_"
            comparison_strategies = [evaluate_fn[6:]]  # clip cc_oe_
            evaluate_fn = EVALUATE_MODE_FN_MAP["cc_oe"]
        elif "us_oe_" in evaluate_fn:
            assert evaluate_fn[:6] == "us_oe_"
            comparison_strategies = [evaluate_fn[6:]]  # clip us_oe_
            evaluate_fn = EVALUATE_MODE_FN_MAP["us_oe"]
        elif "ve_" in evaluate_fn and "oe" in evaluate_fn:
            assert evaluate_fn[:3] == "ve_"
            strategy = "_".join(evaluate_fn.split("_")[3:])
            if len(strategy) == 0:
                strategy = "substring"
            comparison_strategies = [strategy]  # clip oe_

            ve_style = evaluate_fn.split("_")[1]
            evaluate_fn = EVALUATE_MODE_FN_MAP["ve_oe"]
            evaluate_fn = partial(evaluate_fn, ve_style)
        elif "oe_" in evaluate_fn:
            assert evaluate_fn[:3] == "oe_"
            comparison_strategies = [evaluate_fn[3:]]  # clip oe_
            evaluate_fn = EVALUATE_MODE_FN_MAP["oe"]
        elif "us_oe" == evaluate_fn:
            comparison_strategies = [
                # "substring",
                # "fuzzy_gpt-4-0613",
                "fuzzy_gpt-3.5-turbo-1106",
            ]
            evaluate_fn = EVALUATE_MODE_FN_MAP["us_oe"]
        elif "cc_oe" == evaluate_fn:
            comparison_strategies = [
                # "substring",
                # "fuzzy_gpt-4-0613",
                "fuzzy_gpt-3.5-turbo-1106",
            ]
            evaluate_fn = EVALUATE_MODE_FN_MAP["cc_oe"]
        elif "class_oe" in evaluate_fn:
            comparison_strategies = [
                # "substring",
                # "fuzzy_gpt-4-0613",
                "fuzzy_gpt-3.5-turbo-1106",
            ]
            evaluate_fn = EVALUATE_MODE_FN_MAP["class_oe"]
        elif "oe" == evaluate_fn:
            comparison_strategies = [
                # "substring",
                # "fuzzy_gpt-4-0613",
                "fuzzy_gpt-3.5-turbo-1106",
            ]
            evaluate_fn = EVALUATE_MODE_FN_MAP["oe"]
        else:
            assert (
                evaluate_fn in EVALUATE_MODE_FN_MAP.keys()
            ), f"Unsupported mode '{evaluate_fn}'."
            evaluate_fn = EVALUATE_MODE_FN_MAP[evaluate_fn]
            comparison_strategies = None

    all_metrics = []

    for split_name, data in data_splits:
        with Timer() as train_timer:
            metrics = evaluate_fn(
                accelerator,
                model,
                tokenizer,
                get_loader(
                    data,
                    batch_size=batch_size,
                    pin_memory=True,
                    accelerator=accelerator,
                ),
                comparison_strategies=comparison_strategies,
                log_dir=f"{log_dir}/metrics/{dataset}/{split_name}",
            )
        metrics["dataset"] = dataset
        metrics["split"] = split_name
        metrics["ts"] = train_timer.elapsed

        logging.debug(metrics)

        all_metrics.append(metrics)

    return all_metrics

def _dataset_log_name(dataset: str):
    log_name = dataset
    if log_name.startswith("mmlu_oe_offline"):
        _, b = log_name.split(":")
        b = b.split("/")[-1]
        log_name = f"mmlu:{b}"
    elif log_name.startswith("offline"):
        _, b = log_name.split(":")
        b = b.split("/")[-1]
        log_name = f"offline"
    return log_name