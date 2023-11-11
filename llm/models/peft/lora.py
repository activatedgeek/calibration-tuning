from tempfile import TemporaryDirectory
from accelerate import PartialState as AcceleratorState
from peft import TaskType, LoraConfig, PeftModel, get_peft_model

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
    peft_dir=None,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    is_trainable=False,
    adapter_name="default",
    **config_args,
):
    if peft_dir is not None:
        return get_peft_model_from_checkpoint(
            model,
            peft_dir,
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
        **config_args,
    )
    model = get_peft_model(model, peft_config, adapter_name=adapter_name)

    return model


def freezecopy_base_lora_model(model, adapter_name="_ref"):
    assert isinstance(model, PeftModel), f"Unsupported model {type(model)}"

    accelerator = AcceleratorState()

    with TemporaryDirectory() as tdir:
        with accelerator.local_main_process_first():
            if accelerator.is_local_main_process:
                model.save_pretrained(tdir)

        model.load_adapter(tdir, adapter_name=adapter_name)

        for n, p in model.named_parameters():
            if f".{adapter_name}." in n:
                p.requires_grad_(False)
                ## B matrix is already initialized to zero if freezing a fresh LoRA model.
                # if set_zeros:
                #     p.data.fill_(0.0)
