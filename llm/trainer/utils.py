import wandb
from transformers.trainer import TrainerCallback


class WandbConfigUpdateCallback(TrainerCallback):
    def __init__(self, **config):
        self._config = config

    def on_train_begin(self, _args, state, _control, **_):
        if state.is_world_process_zero:
            wandb.config.update(self._config, allow_val_change=True)

            del self._config
