from peft import TaskType, LoraConfig, get_peft_model

from .utils import get_peft_model_from_checkpoint


class use_adapter:
    def __init__(self, model, adapter_name):
        self.model = model
        self.adapter_name = adapter_name
        self.active_adapter = model.active_adapter

    def __enter__(self):
        self.model.set_adapter(self.adapter_name)

    def __exit__(self, *_):
        self.model.set_adapter(self.active_adapter)


def get_lora_model(
    model,
    peft_id_or_dir=None,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    is_trainable=False,
    adapter_name="default",
    **config_args,
):
    if peft_id_or_dir is not None:
        return get_peft_model_from_checkpoint(
            model,
            peft_id_or_dir,
            is_trainable=is_trainable,
            adapter_name=adapter_name,
            **config_args,
        )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        inference_mode=not is_trainable,
        **config_args,
    )
    model = get_peft_model(model, peft_config, adapter_name=adapter_name)

    return model
