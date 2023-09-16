import logging
from tqdm.auto import tqdm
import torch

from .third_party.calibration import calibration
from ..datasets import get_dataset, get_loader
from ..datasets.llm_utils import (
    tokenize_datasets,
    extract_qa_exact,
    prepare_query,
    DataCollatorForSupervisedDataset,
)


@torch.inference_mode()
def evaluate_via_eos(
    accelerator,
    model,
    tokenizer,
    loader,
    query_format="roman_choice",
):
    """
    Assumes all answers are 1 token and end immediately with EOS token.
    """
    device = accelerator.device

    query_token_vec = None
    all_y, all_logits = [], []
    all_unc_y, all_unc_logits = [], []

    for inputs in tqdm(loader, leave=False):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        _, y, logits = extract_qa_exact(tokenizer, inputs, outputs=outputs)

        query_inputs, query_token_vec = prepare_query(
            tokenizer, inputs, outputs, format=query_format
        )
        query_inputs, query_token_vec = loader.collate_fn(
            query_inputs
        ), query_token_vec.to(device)

        query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
        query_outputs = model(**query_inputs)

        _, unc_y, unc_logits = extract_qa_exact(
            tokenizer, query_inputs, outputs=query_outputs
        )

        [
            l.append(v)
            for l, v in zip(
                (all_y, all_logits, all_unc_y, all_unc_logits),
                accelerator.gather_for_metrics((y, logits, unc_y, unc_logits)),
            )
        ]

    all_y, all_logits, all_unc_y, all_unc_logits = [
        torch.cat(l, dim=0) for l in (all_y, all_logits, all_unc_y, all_unc_logits)
    ]

    all_p = all_logits.softmax(dim=-1)
    all_y_hat = all_p.argmax(dim=-1)
    acc = (all_y == all_y_hat).float().mean()
    ece, _ = calibration(
        all_y, all_y_hat, all_p[torch.arange(all_p.size(0)), all_y_hat]
    )

    all_unc_y, all_unc_p = (
        (all_unc_y.unsqueeze(-1) == query_token_vec).long().argmax(dim=-1),
        all_unc_logits[:, query_token_vec].softmax(dim=-1),
    )
    all_unc_y_hat = all_unc_p.argmax(dim=-1)
    unc_acc = (all_unc_y == all_unc_y_hat).float().mean()
    unc_ece, _ = calibration(
        all_unc_y,
        all_unc_y_hat,
        all_unc_p[torch.arange(all_unc_p.size(0)), all_unc_y_hat],
    )

    ## Using confidence scores from "yes" (idx 1) always.
    qa_unc_ece, _ = calibration(all_y, all_y_hat, all_unc_p[:, 1])

    return {
        "N": all_y.size(0),
        "acc": acc.item(),
        "ece": ece,
        "unc_acc": unc_acc.item(),
        "unc_ece": unc_ece,
        "qa_unc_ece": qa_unc_ece,
    }


def evaluate_dataset(
    accelerator,
    model,
    tokenizer,
    dataset,
    evaluate_fn,
    val_data=None,
    test_data=None,
    seed=137,
    batch_size=1,
    num_workers=8,
    data_dir=None,
    eval_kshot=None,
    use_cache=True,
):
    ## FIXME: See https://github.com/huggingface/transformers/issues/25790#issuecomment-1695846805.
    assert batch_size == 1, "Only support batch_size 1. See code comments."

    if dataset is not None:
        with accelerator.main_process_first():
            _extra_args = dict()
            ## NOTE: Conditional to avoid overriding default kshot specification in dataset definition.
            if eval_kshot is not None:
                _extra_args["eval_kshot"] = eval_kshot
            _, val_data, test_data = get_dataset(
                dataset,
                root=data_dir,
                tokenizer=tokenizer,
                seed=seed,
                num_workers=num_workers,
                use_cache=use_cache,
                **_extra_args,
            )
            val_data, test_data = tokenize_datasets(tokenizer, val_data, test_data)
    else:
        assert (val_data is not None) or (
            test_data is not None
        ), "Missing val_data or test_data."

    val_metrics = None
    if val_data is not None:
        val_metrics = evaluate_fn(
            accelerator,
            model,
            tokenizer,
            get_loader(
                val_data,
                batch_size=batch_size,
                collate_fn=DataCollatorForSupervisedDataset(tokenizer),
                accelerator=accelerator,
            ),
        )
        val_metrics["split"] = "validation"

        logging.debug(val_metrics)

    test_metrics = None
    if test_data is not None:
        test_metrics = evaluate_fn(
            accelerator,
            model,
            tokenizer,
            get_loader(
                test_data,
                batch_size=batch_size,
                collate_fn=DataCollatorForSupervisedDataset(tokenizer),
                accelerator=accelerator,
            ),
        )
        test_metrics["split"] = "test"

        logging.debug(test_metrics)

    return val_metrics, test_metrics
