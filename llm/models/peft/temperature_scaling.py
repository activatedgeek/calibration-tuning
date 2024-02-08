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


def get_temperature_scaled_model(
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
