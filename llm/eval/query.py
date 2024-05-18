from tqdm.auto import tqdm
import torch
from collections import OrderedDict
from peft import PeftModel

from ..datasets import (
    LabeledStringDataCollator,
    LMText,
    prepare_uncertainty_query,
    get_token_vec,
)
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
    strategy="substring",
    query_format="roman_choice",
    adapter_name="query",
):
    q_inputs, q_labels, q_token_vec = prepare_uncertainty_query(
        tokenizer,
        lmtext_inputs,
        targets,
        outputs,
        strategy=strategy,
        query_labels=query_labels,
        format=query_format,
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
    grade_strategy=None,
    **_,
):
    eval_data = OrderedDict([("q_logits", []), ("q_labels", [])])
    logits_eval_data = OrderedDict([("logits", []), ("labels", [])])

    # modiste_data = OrderedDict([("output", []), ("example_idx", []), ("orig_example_idx", [])])

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
                choice_vec = get_token_vec(tokenizer, format="choice")

                labels = tokenizer(targets, return_tensors="pt").get("input_ids")[:, 1]
                labels = (
                    (labels.unsqueeze(dim=-1) == choice_vec.unsqueeze(dim=0))
                    .long()
                    .argmax(dim=-1)
                )

                logits = __generations.logits[-1][..., choice_vec]

                [
                    logits_eval_data[k].append(v.cpu())
                    for k, v in zip(
                        logits_eval_data.keys(),
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
            strategy=grade_strategy,
            query_labels=q_labels,
        )

        [
            eval_data[k].append(v.cpu())
            for k, v in zip(
                eval_data.keys(),
                accelerator.gather_for_metrics((q_logits, q_labels)),
            )
        ]

        # [
        #     modiste_data[k].extend(v.cpu().numpy().tolist())
        #     for k, v in extra_inputs.items()
        # ]
        # modiste_data["output"].extend(outputs)

    eval_data = OrderedDict({k: torch.cat(v, dim=0) for k, v in eval_data.items()})

    all_metrics = compute_uncertainty_metrics(
        eval_data.get("q_labels"),
        eval_data.get("q_logits"),
        prefix="unc_",
    )
    all_metrics["acc"] = eval_data.get("q_labels").float().mean(dim=0).item()
    save_metrics_data(eval_data, log_dir=log_dir, filename="query_data.bin")

    logits_eval_data = OrderedDict(
        {k: torch.cat(v, dim=0) for k, v in logits_eval_data.items() if len(v)}
    )
    if logits_eval_data:
        logits_eval_data["choice_vec"] = choice_vec

        metrics = compute_uncertainty_metrics(
            logits_eval_data.get("labels"),
            logits_eval_data.get("logits"),
            prefix="logits_",
        )
        save_metrics_data(logits_eval_data, log_dir=log_dir, filename="logits_data.bin")

        all_metrics.update(metrics)

    # modiste_data["p"] = eval_data.get("q_logits").softmax(dim=-1)[:, 1].cpu().numpy().tolist()
    # save_metrics_data(modiste_data, log_dir=log_dir, filename="modiste_data.bin")

    return all_metrics


@torch.inference_mode()
def evaluate_query_logits(
    accelerator,
    model,
    tokenizer,
    loader,
    log_dir=None,
    T=1.0,
    **_,
):
    eval_data = OrderedDict([("q_logits", []), ("q_labels", [])])

    for q_logits, q_labels in tqdm(loader):
        q_logits = q_logits / T

        [
            eval_data[k].append(v.cpu())
            for k, v in zip(
                eval_data.keys(),
                accelerator.gather_for_metrics((q_logits, q_labels)),
            )
        ]

    eval_data = OrderedDict({k: torch.cat(v, dim=0) for k, v in eval_data.items()})

    all_metrics = compute_uncertainty_metrics(
        eval_data.get("q_labels"),
        eval_data.get("q_logits"),
        prefix="unc_",
    )
    all_metrics["acc"] = eval_data.get("q_labels").float().mean(dim=0).item()

    return all_metrics
