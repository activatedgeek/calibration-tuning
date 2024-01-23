import logging
import os

from ..datasets import get_dataset, get_loader
from .eos import (
    evaluate_contextual_calibration_via_eos,
    evaluate_candidate_via_eos,
    evaluate_via_eos,
)
from .oe import (
    evaluate_oe,
    evaluate_contextual_calibration_oe,
    evaluate_oe_uncertainty_sampling,
    evaluate_verbal_elicitation_oe,
)

EVALUATE_MODE_FN_MAP = {
    "eos": evaluate_via_eos,
    "cc_eos": evaluate_contextual_calibration_via_eos,
    "cand_eos": evaluate_candidate_via_eos,
    "oe": evaluate_oe,
    "us_oe": evaluate_oe_uncertainty_sampling,
    "cc_oe": evaluate_contextual_calibration_oe,
    "ve_oe": evaluate_verbal_elicitation_oe,
}


def evaluate_dataset(
    accelerator,
    model,
    tokenizer,
    dataset,
    train_data=None,
    val_data=None,
    test_data=None,
    seed=137,
    batch_size=1,
    num_workers=8,
    data_dir=None,
    eval_kshot=None,
    use_cache=True,
    prompt_style="choice",
    log_dir=None,
    evaluate_fn="eos",
):
    ## FIXME: See https://github.com/huggingface/transformers/issues/25790#issuecomment-1695846805.
    assert batch_size == 1, "Only support batch_size 1. See code comments."

    if output_row_path is not None:
        os.makedirs(os.path.join(output_row_path, dataset), exist_ok=True)

    if dataset is not None:
        with accelerator.main_process_first():
            _extra_args = dict()
            ## NOTE: Conditional to avoid overriding default kshot specification in dataset definition.
            if eval_kshot is not None:
                _extra_args["eval_kshot"] = eval_kshot
            _train_data, val_data, test_data = get_dataset(
                dataset,
                root=data_dir,
                tokenizer=tokenizer,
                seed=seed,
                num_workers=num_workers,
                use_cache=use_cache,
                prompt_style=prompt_style,
                **_extra_args,
            )
            train_data = _train_data if train_data else None
    else:
        if (val_data is not None) and (test_data is not None):
            logging.warning(f"Missing val_data or test_data.")

    if isinstance(evaluate_fn, str):
        if "cc_oe_" in evaluate_fn:
            assert evaluate_fn[:6] == "cc_oe_"
            comparison_strategies = [evaluate_fn[6:]]  # clip cc_oe_
            evaluate_fn = EVALUATE_MODE_FN_MAP["cc_oe"]
        elif "us_oe_" in evaluate_fn:
            assert evaluate_fn[:6] == "us_oe_"
            comparison_strategies = [evaluate_fn[6:]]  # clip us_oe_
            evaluate_fn = EVALUATE_MODE_FN_MAP["us_oe"]
        elif "oe_" in evaluate_fn:
            assert evaluate_fn[:3] == "oe_"
            comparison_strategies = [evaluate_fn[3:]]  # clip oe_
            evaluate_fn = EVALUATE_MODE_FN_MAP["oe"]
        elif "us_oe" == evaluate_fn:
            comparison_strategies = [
                "substring",
                # "fuzzy_gpt-4-0613",
                "fuzzy_gpt-3.5-turbo-1106",
            ]
            evaluate_fn = EVALUATE_MODE_FN_MAP["us_oe"]
        elif "ve_oe" == evaluate_fn:
            comparison_strategies = [
                "substring",
                # "fuzzy_gpt-4-0613",
                # "fuzzy_gpt-3.5-turbo-1106",
            ]
            evaluate_fn = EVALUATE_MODE_FN_MAP["ve_oe"] 
        elif "cc_oe" == evaluate_fn:
            comparison_strategies = [
                "substring",
                # "fuzzy_gpt-4-0613",
                "fuzzy_gpt-3.5-turbo-1106",
            ]
            evaluate_fn = EVALUATE_MODE_FN_MAP["cc_oe"]
        elif "oe" == evaluate_fn:
            comparison_strategies = [
                "substring",
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

    train_metrics = None

    if train_data:
        train_metrics = evaluate_fn(
            accelerator,
            model,
            tokenizer,
            get_loader(
                train_data,
                batch_size=batch_size,
                pin_memory=True,
                accelerator=accelerator,
                # turn list of dicts into dict of lists
                collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0].keys()},
            ),
            prompt_style=prompt_style,
            comparison_strategies=comparison_strategies,
            output_row_path=os.path.join(log_dir, dataset, "train.csv")
            if log_dir is not None
            else None,
        )
        train_metrics["split"] = "train"

        logging.debug(train_metrics)
    else:
        logging.debug(f"Skipping train data for {dataset}")

    val_metrics = None
    if val_data:
        val_metrics = evaluate_fn(
            accelerator,
            model,
            tokenizer,
            get_loader(
                val_data,
                batch_size=batch_size,
                pin_memory=True,
                accelerator=accelerator,
                collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0].keys()},
            ),
            prompt_style=prompt_style,
            comparison_strategies=comparison_strategies,
            output_row_path=os.path.join(log_dir, dataset, "val.csv")
            if log_dir is not None
            else None,
        )
        val_metrics["split"] = "validation"

        logging.debug(val_metrics)
    else:
        logging.debug(f"Skipping validation data for {dataset}")

    test_metrics = None
    if test_data:
        test_metrics = evaluate_fn(
            accelerator,
            model,
            tokenizer,
            get_loader(
                test_data,
                batch_size=batch_size,
                pin_memory=True,
                accelerator=accelerator,
                collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0].keys()},
            ),
            prompt_style=prompt_style,
            comparison_strategies=comparison_strategies,
            output_row_path=os.path.join(log_dir, dataset, "test.csv")
            if log_dir is not None
            else None,
        )
        test_metrics["split"] = "test"

        logging.debug(test_metrics)
    else:
        logging.debug(f"Skipping test data for {dataset}")

    return val_metrics, test_metrics
