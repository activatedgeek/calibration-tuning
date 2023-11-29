import logging

from ..datasets import get_dataset, get_loader
from .eos import (
    evaluate_contextual_calibration_via_eos,
    evaluate_candidate_via_eos,
    evaluate_via_eos,
)
from .oe import (
    evaluate_oe,
    evaluate_oe_via_substring, 
    evaluate_oe_via_fuzzy_gpt4,
)

EVALUATE_MODE_FN_MAP = {
    "eos": evaluate_via_eos,
    "oe": evaluate_oe,
    "oe_substring": evaluate_oe_via_substring,
    "oe_fuzzy_gpt4": evaluate_oe_via_fuzzy_gpt4,
    "cc_eos": evaluate_contextual_calibration_via_eos,
    "cand_eos": evaluate_candidate_via_eos,
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
    output_row_path=None,
    evaluate_fn="eos",
):
    ## FIXME: See https://github.com/huggingface/transformers/issues/25790#issuecomment-1695846805.
    assert batch_size == 1, "Only support batch_size 1. See code comments."

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
        assert (
            evaluate_fn in EVALUATE_MODE_FN_MAP.keys()
        ), f"Unsupported mode '{evaluate_fn}'."
        evaluate_fn = EVALUATE_MODE_FN_MAP[evaluate_fn]

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
            ),
            prompt_style=prompt_style,
            output_row_path=output_row_path
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
            ),
            prompt_style=prompt_style,
            output_row_path=output_row_path
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
            ),
            prompt_style=prompt_style,
            output_row_path=output_row_path
        )
        test_metrics["split"] = "test"

        logging.debug(test_metrics)
    else:
        logging.debug(f"Skipping test data for {dataset}")

    return val_metrics, test_metrics
