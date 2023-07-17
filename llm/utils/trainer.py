import torch
from transformers import Trainer

from ..datasets.llm_utils import (
    DataCollatorForSupervisedDataset,
    tokenize_for_causal_lm,
)
from .evaluation import extract_aligned_eos_pos, evaluate


class CalibrationTrainer(Trainer):
    def __init__(self, *args, tokenizer=None, beta=0.1, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSupervisedDataset(tokenizer),
        )
        self.beta = beta

    def compute_unc_loss(self, model, inputs, outputs):
        ## TODO: Use sampling for output generation?
        input_ids, shifted_labels, output_ids = (
            inputs.get("input_ids"),
            inputs.get("labels")[..., 1:],
            outputs.logits[..., :-1, :].argmax(dim=-1),
        )

        eos_idx = extract_aligned_eos_pos(self.tokenizer, shifted_labels)

        target_matches = (
            shifted_labels[torch.arange(shifted_labels.size(0)), eos_idx - 1]
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
            for u, r in zip(unc_prompts, target_matches)
        ]
        tokenized_unc_samples = [
            tokenize_for_causal_lm(self.tokenizer, sample) for sample in unc_samples
        ]
        unc_inputs = self.data_collator(tokenized_unc_samples)

        unc_loss = super().compute_loss(model, unc_inputs)

        return unc_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        if self.beta > 0.0:
            unc_loss = self.compute_unc_loss(model, inputs, outputs)
            loss = loss + self.beta * unc_loss

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        custom_metrics = evaluate(
            self.accelerator,
            self.model,
            self.tokenizer,
            self.get_eval_dataloader(eval_dataset),
        )
        custom_metrics = {
            f"{metric_key_prefix}_{k}": v for k, v in custom_metrics.items()
        }
        self.log(custom_metrics)

        metrics.update(custom_metrics)
        return metrics
