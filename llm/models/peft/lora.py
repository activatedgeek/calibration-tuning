from peft import TaskType, LoraConfig, get_peft_model

from .utils import get_peft_model_from_checkpoint


def get_lora_model(model, peft_dir=None, lora_rank=8, lora_alpha=32, lora_dropout=0.1):
    if peft_dir is not None:
        return get_peft_model_from_checkpoint(model, peft_dir)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(model, peft_config)

    return model
