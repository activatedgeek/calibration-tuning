from accelerate import PartialState as AcceleratorState
from peft import PeftModel

from ...utils.trainer import get_last_checkpoint_path


def get_peft_model_from_checkpoint(model, peft_dir):
    accelerator = AcceleratorState()

    peft_dir = get_last_checkpoint_path(peft_dir)

    model = PeftModel.from_pretrained(model, peft_dir, is_trainable=True)

    if accelerator.is_main_process:
        print(f"[INFO]: Loaded PEFT checkpoint from '{peft_dir}'")

    return model
