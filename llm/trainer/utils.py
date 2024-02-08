import wandb
from transformers.trainer import TrainerCallback


class WandbConfigUpdateCallback(TrainerCallback):
    def __init__(self, **config):
        self._config = config

    def on_train_begin(self, _args, state, _control, **_):
        if state.is_world_process_zero:
            wandb.config.update(self._config, allow_val_change=True)

            del self._config


class SchedulerInitCallback(TrainerCallback):
    def __init__(self, scheduler):
        super().__init__()

        self.scheduler = scheduler

    def on_train_begin(self, args, state, _control, **_):
        self.scheduler.setup(
            init_value=args.unc_decay,
            T_max=int(args.unc_decay_ratio * args.max_steps),
            last_epoch=state.global_step,
            eta_min=0.0 if args.loss_mode == "reg" else args.unc_decay,
        )
