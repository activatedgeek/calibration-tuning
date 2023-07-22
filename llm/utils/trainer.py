from dataclasses import dataclass, field
import torch
from transformers import Trainer, TrainingArguments

from ..datasets.llm_utils import (
    DataCollatorForSupervisedDataset,
    tokenize_for_causal_lm,
)
from .evaluation import extract_eos_pos, evaluate_via_eos


@dataclass
class TrainingArguments(TrainingArguments):
    unc_decay: float = field(default=0.1)


class CalibrationTrainer(Trainer):
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSupervisedDataset(tokenizer),
        )

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

        unc_loss = super().compute_loss(model, unc_inputs)

        return unc_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        loss_metrics = {"lm_loss": loss.detach().item()}

        if self.args.unc_decay > 0.0:
            unc_loss = self.compute_unc_loss(model, inputs, outputs)

            total_loss = loss + self.args.unc_decay * unc_loss

            loss_metrics["unc_loss"] = unc_loss.detach().item()
            loss_metrics["total_loss"] = total_loss.detach().item()
        else:
            total_loss = loss

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(loss_metrics)

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
