from .lora import get_lora_model, use_adapter
from .prompt_tuning import get_prompt_tuning_model
from .temperature_scaling import (
    inject_temperature_scaled_model,
    add_temperature_scale_module,
)
from .classifier_head import get_classifier_head

__all__ = [
    "get_prompt_tuning_model",
    "get_lora_model",
    "use_adapter",
    "inject_temperature_scaled_model",
    "add_temperature_scale_module",
    "get_classifier_head",
]
