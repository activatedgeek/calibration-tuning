from tqdm.auto import tqdm
from collections import OrderedDict
import torch
from peft import PeftModel

from ..datasets import LMText, LabeledStringDataCollator, prepare_uncertainty_query
from .common import (
    get_model_generations,
    save_metrics_data,
    compute_uncertainty_metrics,
)


def get_classifier_inputs_via_embedding_model(model, lmtext_inputs, outputs):
    class_inputs = [
        str(LMText.from_({**inp, "target": t}))
        for inp, t in zip(lmtext_inputs, outputs)
    ]
    class_inputs = model.embedding_model.encode(
        class_inputs, convert_to_tensor=True, show_progress_bar=False
    )

    return class_inputs


def get_classifier_inputs(
    accelerator, model, tokenizer, lmtext_inputs, outputs, adapter_name="query"
):
    collate_fn = LabeledStringDataCollator(tokenizer)

    inputs = [{**inp, "target": t} for inp, t in zip(lmtext_inputs, outputs)]
    inputs = {k: v.to(accelerator.device) for k, v in collate_fn(inputs).items()}

    if isinstance(model, PeftModel):
        active_adapter = model.active_adapter
        if adapter_name in model.peft_config:
            model.set_adapter(adapter_name)

    class_inputs = model(**inputs, output_hidden_states=True)

    if isinstance(model, PeftModel):
        model.set_adapter(active_adapter)

    target_layer = getattr(model.classifier_model, "target_layer", -1)
    class_inputs = class_inputs.hidden_states[target_layer][..., -1, :]

    return class_inputs


@torch.inference_mode()
def evaluate_classifier(
    accelerator,
    model,
    tokenizer,
    loader,
    log_dir=None,
    max_new_tokens=None,
    grade_strategy=None,
    **_,
):
    eval_data = OrderedDict([("logits", []), ("labels", [])])

    for inputs in tqdm(loader):
        extra_inputs = {
            k: v for k, v in inputs.items() if k not in LMText.field_names()
        }
        inputs = {k: v for k, v in inputs.items() if k in LMText.field_names()}

        class_inputs = extra_inputs.pop("embedding", None)
        class_labels = inputs.pop("query_label", None)
        outputs = inputs.pop("output", None)
        targets = inputs.pop("target", None)

        inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]

        if outputs is None:
            outputs, _ = get_model_generations(
                accelerator, model, tokenizer, inputs, max_new_tokens=max_new_tokens
            )

        _, class_labels, _ = prepare_uncertainty_query(
            tokenizer,
            inputs,
            targets,
            outputs,
            strategy=grade_strategy,
            query_labels=class_labels,
        )
        class_labels = class_labels.to(accelerator.device)

        if hasattr(model, "embedding_model"):
            if class_inputs is None:
                class_inputs = get_classifier_inputs_via_embedding_model(
                    model, inputs, outputs
                )
        else:
            class_inputs = get_classifier_inputs(
                accelerator, model, tokenizer, inputs, outputs
            )
        class_inputs = class_inputs.to(model.dtype)

        class_logits = model.classifier_model(class_inputs)

        [
            eval_data[k].append(v.cpu())
            for k, v in zip(
                eval_data.keys(),
                accelerator.gather_for_metrics((class_logits, class_labels)),
            )
        ]

    eval_data = OrderedDict({k: torch.cat(v, dim=0) for k, v in eval_data.items()})

    all_metrics = compute_uncertainty_metrics(
        eval_data.get("labels"),
        eval_data.get("logits"),
        prefix="unc_",
    )
    all_metrics["acc"] = eval_data.get("labels").float().mean(dim=0).item()
    save_metrics_data(eval_data, log_dir=log_dir, filename="classifier_data.bin")

    return all_metrics
