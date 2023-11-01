import logging
import os
import torch
import torch.nn as nn
from accelerate import PartialState as AcceleratorState
from transformers.integrations import TrainerCallback

from .utils import get_last_checkpoint_path
from ..utils import setchainattr, getchainattr


WEIGHTS_NAME = "temperature_adapter.bin"


class TemperatureScale(nn.Module):
    PARAMETER_NAME = "log_temperature"

    def __init__(self):
        super().__init__()

        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        t = self.log_temperature.exp()
        return logits / t


def get_temperature_scaled_model(
    model, module_name="lm_head", checkpoint_dir=None, is_trainable=True
):
    assert hasattr(model, module_name), f"{module_name} not found in model."

    accelerator = AcceleratorState()

    temperature_module = TemperatureScale()
    if checkpoint_dir is not None:
        checkpoint_dir = get_last_checkpoint_path(checkpoint_dir)

        if os.path.isfile(f"{checkpoint_dir}/{WEIGHTS_NAME}"):
            ## Remove prefixes while loading to avoid name conflicts.
            state_dict = {
                k.split(".")[-1]: v
                for k, v in torch.load(f"{checkpoint_dir}/{WEIGHTS_NAME}").items()
            }
            temperature_module.load_state_dict(state_dict)

            logging.info(f"Loaded temperature adapter from '{checkpoint_dir}'")

    target_module_names = [
        n for n, _ in model.named_modules() if module_name == n.split(".")[-1]
    ]
    assert (
        len(target_module_names) == 1
    ), f'Ambiguous module name "{module_name}" provided. Found {target_module_names}'

    module_name = target_module_names[0]

    if is_trainable:
        logging.debug(f"Freezing existing model parameters.")
        for _, p in model.named_parameters():
            p.requires_grad_(False)
    else:
        for _, p in temperature_module.parameters():
            p.requires_grad_(False)

    ## Hotpatch temperature module.
    setchainattr(
        model,
        module_name,
        nn.Sequential(
            getchainattr(model, module_name), temperature_module.to(accelerator.device)
        ),
    )

    return model


def save_temperature_scaled_model(model, path):
    ## Assumes scaling used only once, strip of module path for independent reloading.
    state_dict = {
        k.split(".")[-1]: v
        for k, v in model.state_dict().items()
        if TemperatureScale.PARAMETER_NAME in k
    }

    if not len(state_dict):
        logging.warning(
            f"Parameter {TemperatureScale.PARAMETER_NAME} not found. Skipping save."
        )
        return

    os.makedirs(path, exist_ok=True)
    with open(f"{path}/{WEIGHTS_NAME}", "wb") as f:
        torch.save(state_dict, f)


class TemperatureSaveCallback(TrainerCallback):
    def on_save(self, args, state, *_, model=None, **_kwargs):
        if state.is_world_process_zero:
            checkpoint_path = f"{args.output_dir}/checkpoint-{state.global_step}"

            save_temperature_scaled_model(model, checkpoint_path)
