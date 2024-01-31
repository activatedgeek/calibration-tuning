import tqdm
from dataclasses import dataclass, field
import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint

from ..datasets.llm_utils import DataCollatorForSupervisedDataset


__all__ = [
    "WandbConfigUpdateCallback",
    "TrainingArguments",
]


def get_last_checkpoint_path(path):
    if PREFIX_CHECKPOINT_DIR not in path:
        path = get_last_checkpoint(path)

    assert path is not None, f"No checkpoint found in '{path}'."

    return path


class WandbConfigUpdateCallback(TrainerCallback):
    def __init__(self, **config):
        self._config = config

    def on_train_begin(self, _args, state, _control, **_):
        if state.is_world_process_zero:
            import wandb

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


@dataclass
class TrainingArguments(TrainingArguments):
    unc_decay_ratio: float = field(default=1.0)
    unc_decay: float = field(default=0.0)
    unc_normalize: bool = field(default=True)
    loss_mode: str = field(default="reg")
    query_format: str = field(default="roman_choice")
    label_smoothing: float = field(default=0.0)


class ClassificationTrainer(Trainer):
    def __init__(self, base_model, *args, tokenizer=None, test_dataset=None, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSupervisedDataset(tokenizer),
        )

        self.base_model = base_model
        self.test_dataset = test_dataset

    def compute_loss(self, model, inputs, return_outputs=False):
        _, outputs = super().compute_loss(self.base_model, inputs, return_outputs=True)

        input_ids, labels, output_ids = (
            inputs.get("input_ids"),
            inputs.get("labels")[..., 1:],
            outputs.logits[..., :-1, :].argmax(dim=-1).detach(),
        )
        attn_mask = inputs.get("attention_mask")

        eos_idx = labels.eq(self.tokenizer.eos_token_id).nonzero()[
            labels.eq(self.tokenizer.eos_token_id).sum(dim=-1).cumsum(dim=0) - 1
        ][:, -1]

        targets = (
            labels[torch.arange(labels.size(0)), eos_idx - 1]
            == output_ids[torch.arange(output_ids.size(0)), eos_idx - 1]
        )

        response_ids = input_ids.clone()
        response_ids[torch.arange(input_ids.size(0)), eos_idx] = output_ids[
            torch.arange(output_ids.size(0)), eos_idx - 1
        ]

        response_ids[
            torch.arange(input_ids.size(0)), eos_idx + 1
        ] = self.tokenizer.pad_token_id

        labels = torch.tensor(targets, device=targets.device).long()

        loss = model(
            input_ids=response_ids,
            attention_mask=attn_mask,
            labels=labels,
        ).loss

        loss_metrics = {
            "loss": loss.detach().item(),
        }

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(loss_metrics)

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = {}

        eval_loader = self.get_eval_dataloader(eval_dataset)

        device = self.accelerator.device
        loss = 0.0
        for inputs in tqdm.tqdm(eval_loader, leave=False):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            loss += self.compute_loss(self.model, inputs)
        loss /= len(eval_loader)

        val_metrics = {f"{metric_key_prefix}_loss": loss.detach().item()}

        self.log(val_metrics)
        metrics.update(val_metrics)

        test_loader = self.get_test_dataloader(self.test_dataset)

        loss = 0.0
        for inputs in tqdm.tqdm(test_loader, leave=False):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            loss += self.compute_loss(self.model, inputs)
        loss /= len(test_loader)

        test_metrics = {f"test_loss": loss.detach().item()}
        self.log(test_metrics)
        metrics.update(test_metrics)

        return metrics
