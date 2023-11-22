import logging
from peft import PeftModel

from ...utils.trainer import get_last_checkpoint_path


def get_peft_model_from_checkpoint(
    model, peft_dir, is_trainable=True, adapter_name="default", **config_args
):
    peft_dir = get_last_checkpoint_path(peft_dir)

    if isinstance(model, PeftModel):
        model.load_adapter(
            peft_dir,
            is_trainable=is_trainable,
            adapter_name=adapter_name,
            **config_args,
        )
    else:
        model = PeftModel.from_pretrained(
            model,
            peft_dir,
            is_trainable=is_trainable,
            adapter_name=adapter_name,
            **config_args,
        )

    logging.info(f"Loaded PEFT adapter '{adapter_name}' checkpoint from '{peft_dir}'")

    return model
