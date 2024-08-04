import os
import logging
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from transformers import GenerationConfig
from peft import PeftModel

from ..datasets import LabeledStringDataCollator
from ..datasets.llm_utils_oe import sanitize_generations
from .third_party.calibration import calibration


DATA_FILE_NAME = "data.bin"


def save_metrics_data(data, log_dir=None, filename=DATA_FILE_NAME):
    if log_dir is None:
        return

    os.makedirs(log_dir, exist_ok=True)

    torch.save(data, f"{log_dir}/{filename}")

    logging.info(f'Metrics data saved to "{log_dir}/{filename}".')


def compute_auroc(labels, probs, multi_class="ovr", **kwargs):
    one_hot_labels = (
        F.one_hot(labels, num_classes=probs.size(-1)) if labels.ndim == 1 else labels
    )

    try:
        auroc = roc_auc_score(one_hot_labels, probs.float(), multi_class=multi_class, **kwargs)
    except ValueError:
        auroc = float("nan")
        logging.exception("AUROC calculation failed.", exc_info=True)

    return auroc


def compute_uncertainty_metrics(labels, logits, prefix=""):
    """
    Arguments:
        labels: Shape (N,)
        logits: Shape (N, 2)
    """
    p = logits.softmax(dim=-1)

    pred = p.argmax(dim=-1)
    acc = (pred == labels).float().mean(dim=0)

    ece, _ = calibration(
        labels,
        pred,
        p[torch.arange(p.size(0)), pred].float(),
    )

    auroc = compute_auroc(labels, p)

    return {
        "N": labels.size(0),
        f"{prefix}acc": acc.item(),
        f"{prefix}auroc": auroc,
        f"{prefix}ece": ece,
    }


def get_model_generations(
    accelerator,
    model,
    tokenizer,
    lmtext_inputs,
    max_new_tokens=None,
    adapter_name="default",
):
    config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_logits=True,
        output_hidden_states=True,
    )

    if max_new_tokens is None:
        logging.warning(f"max_new_tokens is None.")

    collate_fn = LabeledStringDataCollator(tokenizer)

    inputs = collate_fn(lmtext_inputs)
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    if isinstance(model, PeftModel):
        active_adapter = model.active_adapter
        model.set_adapter(adapter_name)

    outputs = model.generate(**inputs, generation_config=config)

    if isinstance(model, PeftModel):
        model.set_adapter(active_adapter)

    str_outputs = tokenizer.batch_decode(
        outputs.sequences[:, inputs.get("input_ids").size(-1) :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    str_outputs = sanitize_generations(str_outputs)

    return str_outputs, outputs
