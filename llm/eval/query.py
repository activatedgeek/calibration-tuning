from tqdm.auto import tqdm
import torch
from collections import OrderedDict
from peft import PeftModel

from ..datasets import LabeledStringDataCollator, LMText, prepare_uncertainty_query
from .common import (
    get_model_generations,
    compute_uncertainty_metrics,
    save_metrics_data,
)


def get_query_generations(
    accelerator,
    model,
    tokenizer,
    lmtext_inputs,
    targets,
    outputs,
    query_labels=None,
    query_format="roman_choice",
    adapter_name="query",
):
    q_inputs, q_labels, q_token_vec = prepare_uncertainty_query(
        tokenizer,
        lmtext_inputs,
        targets,
        outputs,
        strategy="substring",
        format=query_format,
        query_labels=query_labels,
    )

    collate_fn = LabeledStringDataCollator(tokenizer)

    q_inputs = collate_fn(q_inputs)
    q_inputs = {k: v.to(accelerator.device) for k, v in q_inputs.items()}

    if isinstance(model, PeftModel):
        active_adapter = model.active_adapter
        if adapter_name in model.peft_config:
            model.set_adapter(adapter_name)

    q_generation_outputs = model(**q_inputs)

    if isinstance(model, PeftModel):
        model.set_adapter(active_adapter)

    q_logits = q_generation_outputs.logits[..., -1, q_token_vec]

    if hasattr(model, "query_temperature_model"):
        q_logits = model.query_temperature_model(q_logits)

    return q_logits, q_labels


@torch.inference_mode()
def evaluate_query(
    accelerator,
    model,
    tokenizer,
    loader,
    log_dir=None,
    max_new_tokens=None,
    **_,
):
    query_eval_data = OrderedDict([("q_logits", []), ("q_labels", [])])
    eval_data = OrderedDict([("logits", []), ("labels", [])])

    for inputs in tqdm(loader):
        extra_inputs = {
            k: v for k, v in inputs.items() if k not in LMText.field_names()
        }
        inputs = {k: v for k, v in inputs.items() if k in LMText.field_names()}

        outputs = inputs.pop("output", None)
        targets = inputs.pop("target", None)
        q_labels = inputs.pop("query_label", None)

        inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]

        if outputs is None:
            outputs, __generations = get_model_generations(
                accelerator, model, tokenizer, inputs, max_new_tokens=max_new_tokens
            )

            ## Token-level metrics only for single-token generation.
            if max_new_tokens == 1:
                logits = __generations.logits[-1]
                labels = tokenizer(targets, return_tensors="pt").get("input_ids")[:, 1]

                [
                    eval_data[k].append(v.cpu())
                    for k, v in zip(
                        eval_data.keys(),
                        accelerator.gather_for_metrics((logits, labels)),
                    )
                ]

        q_logits, q_labels = get_query_generations(
            accelerator,
            model,
            tokenizer,
            inputs,
            targets,
            outputs,
            query_labels=q_labels,
        )

        [
            query_eval_data[k].append(v.cpu())
            for k, v in zip(
                query_eval_data.keys(),
                accelerator.gather_for_metrics((q_logits, q_labels)),
            )
        ]

    query_eval_data = OrderedDict(
        {k: torch.cat(v, dim=0) for k, v in query_eval_data.items()}
    )

    all_metrics = compute_uncertainty_metrics(
        query_eval_data.get("q_labels"),
        query_eval_data.get("q_logits"),
        prefix="unc_",
    )
    save_metrics_data(query_eval_data, log_dir=log_dir, filename="query_metrics.bin")

    eval_data = OrderedDict(
        {k: torch.cat(v, dim=0) for k, v in eval_data.items() if len(v)}
    )
    if eval_data:
        metrics = compute_uncertainty_metrics(
            eval_data.get("labels"),
            eval_data.get("logits"),
            prefix="logits_",
        )
        save_metrics_data(eval_data, log_dir=log_dir, filename="logit_metrics.bin")

        all_metrics.update(metrics)

    return all_metrics