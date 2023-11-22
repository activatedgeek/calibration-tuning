import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM


WEIGHTS_NAME = "temperature_adapter.bin"


class TemperatureScaledLlamaForCausalLM(LlamaForCausalLM):
    PARAMETER_NAME = "log_temperature"

    def __init__(self, *args, temp_scaling=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.temp_scaling = temp_scaling

        self.register_parameter(
            self.PARAMETER_NAME,
            nn.Parameter(torch.zeros(1), requires_grad=False),
        )

    def forward(self, *args, scale_temp=None, **kwargs):
        outputs = super().forward(*args, **kwargs)

        should_scale = scale_temp if scale_temp is not None else self.temp_scaling

        if should_scale:
            T = getattr(self, self.PARAMETER_NAME).exp()
            outputs.logits = outputs.logits / T

        return outputs


def prepare_model_for_temperature_scaling(model):
    for n, p in model.named_parameters():
        if TemperatureScaledLlamaForCausalLM.PARAMETER_NAME in n:
            p.requires_grad_(True)
        else:
            if p.requires_grad:
                p.requires_grad_(False)


def save_temperature_scaled_model(model, output_dir):
    ## Assumes scaling used only once, strip of module path for independent reloading.
    state_dict = {
        k: v
        for k, v in model.state_dict().items()
        if TemperatureScaledLlamaForCausalLM.PARAMETER_NAME in k
    }

    if not len(state_dict):
        logging.warning(
            f"Parameter {TemperatureScaledLlamaForCausalLM.PARAMETER_NAME} not found. Skipping save."
        )
        return

    with open(f"{output_dir}/{WEIGHTS_NAME}", "wb") as f:
        torch.save(state_dict, f)

    logging.debug(state_dict)
