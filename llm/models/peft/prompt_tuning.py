from peft import TaskType, PromptTuningConfig, PromptTuningInit, get_peft_model

from .utils import get_peft_model_from_checkpoint


def get_prompt_tuning_model(model, peft_dir=None, num_tokens=8):
    if peft_dir is not None:
        return get_peft_model_from_checkpoint(model, peft_dir)

    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=num_tokens,
    )
    model = get_peft_model(model, peft_config)

    return model
