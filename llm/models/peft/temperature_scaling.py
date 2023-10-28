import logging
import os
import torch
import torch.nn as nn
from accelerate import PartialState as AcceleratorState

from .utils import get_last_checkpoint_path


WEIGHTS_NAME = "temperature_adapter.bin"


class TemperatureScale(nn.Module):
    PARAMETER_NAME = "log_temperature"

    def __init__(self):
        super().__init__()

        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        t = self.log_temperature.exp()
        return logits / t


def get_temperature_scaled_model(model, module_name="lm_head", checkpoint_dir=None):
    assert hasattr(model, module_name), f"{module_name} not found in model."

    accelerator = AcceleratorState()

    temperature_module = TemperatureScale()
    if checkpoint_dir is not None:
        checkpoint_dir = get_last_checkpoint_path(checkpoint_dir)
        ## Remove prefixes while loading to avoid name conflicts.
        state_dict = {
            k.split(".")[-1]: v
            for k, v in torch.load(f"{checkpoint_dir}/{WEIGHTS_NAME}").items()
        }
        temperature_module.load_state_dict(state_dict)

        logging.info(f"Loaded temperature adapter from {checkpoint_dir}")

    new_module = nn.Sequential(
        getattr(model, module_name), temperature_module.to(accelerator.device)
    )

    setattr(model, module_name, new_module)

    return model


def save_temperature_scaled_model(model, path):
    ## Assumes scaling used only once, strip of module path for independent reloading.
    state_dict = {
        k.split(".")[-1]: v
        for k, v in model.state_dict().items()
        if TemperatureScale.PARAMETER_NAME in k
    }

    os.makedirs(path, exist_ok=True)

    with open(f"{path}/{WEIGHTS_NAME}", "wb") as f:
        torch.save(state_dict, f)
