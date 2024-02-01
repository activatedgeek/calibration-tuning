import logging
from tqdm.auto import tqdm
import torch
from peft import PeftModel
from sklearn.metrics import roc_auc_score

from ..datasets import LabeledStringDataCollator
from ..datasets.llm_utils_oe import prepare_oe_uncertainty_query
from ..datasets.llm_utils import (
    DataCollatorForSupervisedDataset,
    get_token_vec,
    prepare_batch,
    extract_qa_exact,
)
from .third_party.calibration import calibration


@torch.inference_mode()
def evaluate_via_eos(
    accelerator,
    model,
    tokenizer,
    loader,
    query_format="roman_choice",
    **_,
):
    collate_fn = LabeledStringDataCollator(tokenizer)

    all_q_labels, all_q_logits = [], []

    for inputs in tqdm(loader):
        inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
        targets = [inp.pop("target") for inp in inputs]

        generation_inputs = {
            k: v.to(accelerator.device) for k, v in collate_fn(inputs).items()
        }

        if isinstance(model, PeftModel):
            model.set_adapter("default")

        generation_outputs = model(**generation_inputs)
        logits = generation_outputs.logits[:, -1, :]

        generations = tokenizer.batch_decode(
            logits.argmax(dim=-1),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        q_inputs, q_labels, q_token_vec = prepare_oe_uncertainty_query(
            tokenizer,
            inputs,
            targets,
            generations,
            strategy="substring",
            format=query_format,
        )
        q_labels = q_labels.to(accelerator.device)

        q_generation_inputs = {
            k: v.to(accelerator.device) for k, v in collate_fn(q_inputs).items()
        }

        if isinstance(model, PeftModel) and "query" in model.peft_config:
            model.set_adapter("query")

        q_generation_outputs = model(**q_generation_inputs)

        q_logits = q_generation_outputs.logits[..., -1, :]

        [
            l.append(v)
            for l, v in zip(
                (all_q_labels, all_q_logits),
                accelerator.gather_for_metrics((q_labels, q_logits)),
            )
        ]

    all_q_labels = torch.cat(all_q_labels, dim=0)
    all_q_p = torch.cat(all_q_logits, dim=0)[:, q_token_vec].softmax(dim=-1)

    acc = all_q_labels.float().mean(dim=0)
    all_q_pred = all_q_p.argmax(dim=-1)
    q_acc = (all_q_pred == all_q_labels).float().mean(dim=0)

    q_ece, _ = calibration(
        all_q_labels,
        all_q_pred,
        all_q_p[torch.arange(all_q_p.size(0)), all_q_pred],
    )

    try:
        q_auroc = roc_auc_score(
            all_q_labels.cpu(),
            all_q_p[torch.arange(all_q_p.size(0)), all_q_pred].cpu(),
        )
    except ValueError:
        logging.warning(f"AUROC calculation failed.")
        q_auroc = float("nan")

    ece, _ = calibration(
        all_q_labels,
        all_q_pred,
        all_q_p[torch.arange(all_q_p.size(0)), 1],  ## corresponds to "yes"
    )

    return {
        "N": all_q_labels.size(0),
        f"acc": acc.item(),
        f"unc_acc": q_acc.item(),
        f"unc_auroc": q_auroc,
        f"unc_ece": q_ece,
        f"ece": ece,
    }


@torch.inference_mode()
def evaluate_contextual_calibration_via_eos(
    accelerator,
    model,
    tokenizer,
    loader,
    prompt_style="choice",
    **_,
):
    """
    Assumes all answers are 1 token and end immediately with EOS token.
    """

    assert prompt_style == "choice"

    device = accelerator.device
    collate_fn = DataCollatorForSupervisedDataset(tokenizer)

    if isinstance(model, PeftModel):
        model.set_adapter("default")

    all_y, all_logits = [], []
    all_platt_logits = []

    for raw_inputs in tqdm(loader, leave=False):
        platt_logits = []

        for cf_str in [
            "Question: N/A",
            "Question: ",
            f"Question: {tokenizer.pad_token}",
        ]:
            calib_inputs = {
                **raw_inputs,
                "context": [cf_str] * len(raw_inputs["context"]),
            }
            calib_inputs = prepare_batch(tokenizer, calib_inputs)
            calib_inputs = collate_fn(calib_inputs)

            calib_inputs = {k: v.to(device) for k, v in calib_inputs.items()}
            calib_outputs = model(**calib_inputs)

            _, _, _c_logits = extract_qa_exact(
                tokenizer, calib_inputs, outputs=calib_outputs
            )

            platt_logits.append(_c_logits)

        ## Ensemble over context-free strings.
        platt_logits = torch.stack(platt_logits).mean(dim=0)

        inputs = prepare_batch(tokenizer, raw_inputs)
        inputs = collate_fn(inputs)

        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        _, y, logits = extract_qa_exact(tokenizer, inputs, outputs=outputs)

        [
            l.append(v)
            for l, v in zip(
                (all_y, all_logits, all_platt_logits),
                accelerator.gather_for_metrics((y, logits, platt_logits)),
            )
        ]

    all_y, all_logits, all_platt_logits = [
        torch.cat(l, dim=0) for l in (all_y, all_logits, all_platt_logits)
    ]

    all_p = all_logits.softmax(dim=-1)
    all_y_hat = all_p.argmax(dim=-1)
    acc = (all_y == all_y_hat).float().mean()
    logits_ece, _ = calibration(
        all_y, all_y_hat, all_p[torch.arange(all_p.size(0)), all_y_hat]
    )

    all_cal_logits = all_logits - all_platt_logits
    all_cal_p = all_cal_logits.softmax(dim=-1)
    all_cal_y_hat = all_cal_p.argmax(dim=-1)
    cal_acc = (all_y == all_cal_y_hat).float().mean()
    cal_ece, _ = calibration(
        all_y, all_y_hat, all_cal_p[torch.arange(all_cal_p.size(0)), all_cal_y_hat]
    )

    return {
        "N": all_y.size(0),
        "acc": acc.item(),
        "logits_ece": logits_ece,
        "cal_acc": cal_acc.item(),
        "cal_ece": cal_ece,
    }


@torch.inference_mode()
def evaluate_candidate_via_eos(
    accelerator,
    model,
    tokenizer,
    loader,
    prompt_style="choice",
    **_,
):
    """
    Assumes all answers are 1 token and end immediately with EOS token.
    """

    assert prompt_style == "choice"

    device = accelerator.device
    collate_fn = DataCollatorForSupervisedDataset(tokenizer)

    if isinstance(model, PeftModel):
        model.set_adapter("default")

    all_y, all_logits = [], []

    for inputs in tqdm(loader, leave=False):
        inputs = prepare_batch(tokenizer, inputs)
        inputs = collate_fn(inputs)

        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        _, y, logits = extract_qa_exact(tokenizer, inputs, outputs=outputs)

        qa_token_vec = get_token_vec(tokenizer, format="mcq").to(y.device)
        y, logits = (
            (y.unsqueeze(-1) == qa_token_vec).long().argmax(dim=-1),
            logits[:, qa_token_vec],
        )

        [
            l.append(v)
            for l, v in zip(
                (all_y, all_logits),
                accelerator.gather_for_metrics((y, logits)),
            )
        ]

    all_y, all_logits = [torch.cat(l, dim=0) for l in (all_y, all_logits)]

    all_p = all_logits.softmax(dim=-1)
    all_y_hat = all_p.argmax(dim=-1)
    acc = (all_y == all_y_hat).float().mean()
    logits_ece, _ = calibration(
        all_y, all_y_hat, all_p[torch.arange(all_p.size(0)), all_y_hat]
    )

    return {
        "N": all_y.size(0),
        "acc": acc.item(),
        "logits_ece": logits_ece,
    }
