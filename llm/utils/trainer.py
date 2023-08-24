from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from transformers.integrations import TrainerCallback
from transformers.training_args import TrainingArguments

from ..datasets.llm_utils import (
    DataCollatorForSupervisedDataset,
    tokenize_for_causal_lm,
)
from .evaluation import extract_eos_pos, evaluate_via_eos
from .scheduler import AnyCosineScheduler


__all__ = [
    "WandbConfigUpdateCallback",
    "TrainingArguments",
    "CalibrationTrainer",
]


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
        )


@dataclass
class TrainingArguments(TrainingArguments):
    unc_decay_ratio: float = field(default=1.0)
    unc_decay: float = field(default=0.0)
    unc_normalize: bool = field(default=True)
    loss_mode: str = field(default="reg")


class CalibrationTrainer(Trainer):
    def __init__(self, *args, tokenizer=None, test_dataset=None, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSupervisedDataset(tokenizer),
        )

        self.test_dataset = test_dataset

        _no_token = self.tokenizer("no").input_ids
        _yes_token = self.tokenizer("yes").input_ids

        ## NOTE: Assumes yes/no are single token (length 2 including BOS token).
        assert len(_no_token) == 2, f'Cannot handle "no" token {_no_token} yet.'
        assert len(_yes_token) == 2, f'Cannot handle "yes" token {_yes_token} yet.'

        self.uq_ans_token_vec = torch.tensor([_no_token[-1], _yes_token[-1]])

        self.unc_decay = AnyCosineScheduler()
        self.add_callback(SchedulerInitCallback(self.unc_decay))

    def compute_unc_loss(self, model, inputs, outputs):
        input_ids, labels, output_ids = (
            inputs.get("input_ids"),
            inputs.get("labels")[..., 1:],
            outputs.logits[..., :-1, :].argmax(dim=-1).detach(),
        )

        eos_idx = extract_eos_pos(self.tokenizer, labels)

        targets = (
            labels[torch.arange(labels.size(0)), eos_idx - 1]
            == output_ids[torch.arange(output_ids.size(0)), eos_idx - 1]
        )

        response_ids = input_ids.clone()
        response_ids[torch.arange(input_ids.size(0)), eos_idx] = output_ids[
            torch.arange(output_ids.size(0)), eos_idx - 1
        ]
        unc_prompts = self.tokenizer.batch_decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        unc_samples = [
            {
                "source": u + "\n" + "Is the proposed answer correct? ",
                "target": f"{'yes' if r else 'no'}{self.tokenizer.eos_token}",
            }
            for u, r in zip(unc_prompts, targets)
        ]
        tokenized_unc_samples = [
            tokenize_for_causal_lm(self.tokenizer, sample) for sample in unc_samples
        ]
        unc_inputs = self.data_collator(tokenized_unc_samples)

        if self.args.unc_normalize:
            unc_labels = unc_inputs.pop("labels")
            unc_eos_idx = extract_eos_pos(self.tokenizer, unc_labels)

            unc_logits = model(**unc_inputs).logits[
                torch.arange(unc_labels.size(0)), unc_eos_idx - 1, :
            ]
            norm_unc_logits = unc_logits[..., self.uq_ans_token_vec]

            unc_labels = unc_labels[torch.arange(unc_labels.size(0)), unc_eos_idx - 1]
            norm_unc_labels = (
                (unc_labels.unsqueeze(-1) == self.uq_ans_token_vec)
                .long()
                .argmax(dim=-1)
            )

            norm_unc_loss = F.cross_entropy(
                norm_unc_logits, norm_unc_labels.to(norm_unc_logits.device)
            )

            return norm_unc_loss

        unc_loss = super().compute_loss(model, unc_inputs)

        return unc_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        unc_loss = (
            self.compute_unc_loss(model, inputs, outputs)
            if self.unc_decay.value > 0.0
            else torch.tensor(0.0)
        )

        if self.args.loss_mode == "reg":
            total_loss = loss + self.unc_decay.value * unc_loss
        elif self.args.loss_mode == "cvx_comb":
            total_loss = (
                1 - self.unc_decay.value
            ) * loss + self.unc_decay.value * unc_loss
        else:
            raise NotImplementedError

        loss_metrics = {
            "lm_loss": loss.detach().item(),
            "unc_loss": unc_loss.detach().item(),
            "total_loss": total_loss.detach().item(),
            "unc_decay": self.unc_decay.value,
        }

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(loss_metrics)

        self.unc_decay.step()

        return (total_loss, outputs) if return_outputs else total_loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        val_metrics = evaluate_via_eos(
            self.accelerator,
            self.model,
            self.tokenizer,
            self.get_eval_dataloader(eval_dataset),
        )
        val_metrics = {f"{metric_key_prefix}_{k}": v for k, v in val_metrics.items()}
        self.log(val_metrics)
        metrics.update(val_metrics)

        test_metrics = evaluate_via_eos(
            self.accelerator,
            self.model,
            self.tokenizer,
            self.get_test_dataloader(self.test_dataset),
        )
        test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
        self.log(test_metrics)
        metrics.update(test_metrics)

        return metrics
