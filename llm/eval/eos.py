from tqdm.auto import tqdm
import torch
from peft import PeftModel

from ..datasets.llm_utils import (
    DataCollatorForSupervisedDataset,
    prepare_batch,
    extract_qa_exact,
    extract_oe_inputs,
    prepare_query,
)
from .third_party.calibration import calibration

@torch.inference_mode()
def evaluate_via_eos(
    accelerator,
    model,
    tokenizer,
    loader,
    prompt_style="oe",
    query_format="roman_choice",
):
    """
    Assumes all answers are 1 token and end immediately with EOS token.
    """
    device = accelerator.device
    collate_fn = DataCollatorForSupervisedDataset(tokenizer)

    query_token_vec = None
    all_y, all_logits = [], []
    all_unc_y, all_unc_logits = [], []

    for inputs in tqdm(loader, leave=False):
        inputs = prepare_batch(tokenizer, inputs, prompt_style=prompt_style)
        inputs = collate_fn(inputs)

        if isinstance(model, PeftModel):
            model.set_adapter("default")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        _, y, logits = extract_qa_exact(tokenizer, inputs, outputs=outputs)

        query_inputs, query_token_vec = prepare_query(
            tokenizer, inputs, outputs, format=query_format
        )
        query_inputs = collate_fn(query_inputs)

        if isinstance(model, PeftModel) and "query" in model.peft_config:
            model.set_adapter("query")

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

    query_token_vec = query_token_vec.to(device)
    all_y, all_logits, all_unc_y, all_unc_logits = [
        torch.cat(l, dim=0) for l in (all_y, all_logits, all_unc_y, all_unc_logits)
    ]

    ## Renormalize over options.
    # from ..datasets.llm_utils import get_token_vec
    #
    # qa_token_vec = get_token_vec(tokenizer, format="mcq").to(all_y.device)
    # all_y, all_p = (
    #     (all_y.unsqueeze(-1) == qa_token_vec).long().argmax(dim=-1),
    #     all_logits[:, qa_token_vec].softmax(dim=-1),
    # )

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


@torch.inference_mode()
def evaluate_contextual_calibration_via_eos(
    accelerator,
    model,
    tokenizer,
    loader,
    prompt_style="choice"
):
    """
    Assumes all answers are 1 token and end immediately with EOS token.
    """
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
    ece, _ = calibration(
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
        "ece": ece,
        "cal_acc": cal_acc.item(),
        "cal_ece": cal_ece,
    }
