import os
import logging
import torch
import torch.nn as nn

from .utils import get_last_checkpoint_path


class TemperatureScale(nn.Module):
    NAME = "log_temperature"

    def __init__(self):
        super().__init__()

        self.register_parameter(
            self.NAME,
            nn.Parameter(torch.tensor(0.0)),
        )

    def forward(self, inputs):
        return inputs / self.get_parameter(self.NAME).exp()


def add_temperature_scale_module(
    model,
    peft_dir=None,
    is_trainable=False,
    ref_module_name="lm_head",
    target_module_name="temperature_head",
    register_hook=False,
):
    ref_module = [
        module for key, module in model.named_modules() if ref_module_name in key
    ]
    assert len(ref_module), f"Reference module '{ref_module_name}' not found."

    ref_module = ref_module[0]
    device, dtype = [(p.device, p.dtype) for _, p in ref_module.named_parameters()][0]

    temperature_module = (
        TemperatureScale().to(device).to(dtype).requires_grad_(is_trainable)
    )

    if peft_dir is not None:
        peft_dir = get_last_checkpoint_path(peft_dir)

        if os.path.isfile(f"{peft_dir}/{target_module_name}.bin"):
            temperature_module.load_state_dict(
                torch.load(f"{peft_dir}/{target_module_name}.bin")
            )

        logging.info(f"Loaded temperature scale checkpoint from '{peft_dir}'.")

    if is_trainable:
        model.requires_grad_(False)

    model.register_module(target_module_name, temperature_module)

    if register_hook:
        model.get_submodule(ref_module_name).register_forward_hook(
            lambda _module, _inputs, outputs: model.get_submodule(target_module_name)(
                outputs
            )
        )

    return model


def inject_temperature_scaled_model(
    model, peft_dir=None, is_trainable=False, target_module_name="lm_head"
):
    if peft_dir is not None:
        peft_dir = get_last_checkpoint_path(peft_dir)

        ## TODO: load checkpoint

    for key, mod in model.named_modules():
        if key.endswith(target_module_name):
            device, dtype = [(p.device, p.dtype) for _, p in mod.named_parameters()][0]

            temperature_module = (
                TemperatureScale().to(device).to(dtype).requires_grad_(is_trainable)
            )

            new_module = nn.Sequential(
                mod,
                temperature_module,
            )

            parent = model.get_submodule(".".join(key.split(".")[:-1]))
            target_name = key.split(".")[-1]
            setattr(parent, target_name, new_module)

            logging.info(f"Injected temperature scaling module at {key}.")

    return model
