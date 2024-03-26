import os
import logging
from peft import PeftModel
from transformers.trainer import (
    PREFIX_CHECKPOINT_DIR,
    get_last_checkpoint as __get_last_checkpoint,
)


def get_last_checkpoint_path(path):
    if PREFIX_CHECKPOINT_DIR not in path:
        path = __get_last_checkpoint(path)

    assert path is not None, f"No checkpoint found in '{path}'."

    return path


def get_peft_model_from_checkpoint(
    model,
    peft_id_or_dir,
    is_trainable=True,
    adapter_name="default",
    **config_args,
):
    if os.path.isdir(peft_id_or_dir):
        peft_id_or_dir = get_last_checkpoint_path(peft_id_or_dir)

    if isinstance(model, PeftModel):
        model.load_adapter(
            peft_id_or_dir,
            is_trainable=is_trainable,
            adapter_name=adapter_name,
            **config_args,
        )
    else:
        model = PeftModel.from_pretrained(
            model,
            peft_id_or_dir,
            is_trainable=is_trainable,
            adapter_name=adapter_name,
            **config_args,
        )

    logging.info(
        f"Loaded PEFT adapter '{adapter_name}' checkpoint from '{peft_id_or_dir}'"
    )

    return model
