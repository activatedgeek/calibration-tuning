import os
import logging
import torch
import torch.nn as nn

from .utils import get_last_checkpoint_path


class TemperatureScale(nn.Module):
    def __init__(self):
        super().__init__()

        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        return inputs / self.log_temperature.exp()


def get_temperature_head(
    checkpoint_dir=None, is_trainable=False, weights_name="temperature_head.bin"
):

    temperature_model = TemperatureScale()

    if checkpoint_dir is not None:
        checkpoint_dir = get_last_checkpoint_path(checkpoint_dir)

        if os.path.isfile(f"{checkpoint_dir}/{weights_name}"):
            temperature_model.load_state_dict(
                torch.load(f"{checkpoint_dir}/{weights_name}")
            )

            logging.info(
                f"Loaded temperature model checkpoint from '{checkpoint_dir}'."
            )

    if is_trainable:
        temperature_model = temperature_model.train().requires_grad_(True)
    else:
        temperature_model = temperature_model.eval().requires_grad_(False)

    return temperature_model


def get_temperature_scale_model(model, target_module_name="lm_head", **kwargs):

    for key, mod in model.named_modules():
        if key.endswith(target_module_name):
            device = [p.device for _, p in mod.named_parameters()][0]

            temperature_model = get_temperature_head(**kwargs).to(device)

            new_module = nn.Sequential(
                mod,
                temperature_model,
            )

            parent = model.get_submodule(".".join(key.split(".")[:-1]))
            target_name = key.split(".")[-1]
            setattr(parent, target_name, new_module)

            logging.info(f"Added temperature scaling module to {key}.")

    return model
