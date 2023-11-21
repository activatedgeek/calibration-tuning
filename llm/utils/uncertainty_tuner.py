from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
from transformers import Trainer
from transformers.training_args import TrainingArguments

from ..datasets.llm_utils import (
    DataCollatorForSupervisedDataset,
    extract_qa_exact,
    prepare_query,
    IGNORE_LABEL,
)
from ..eval import evaluate_dataset


class UncertaintyTuner(Trainer):
    @dataclass
    class Args(TrainingArguments):
        query_format: str = field(default="roman_choice")
        unc_normalize: bool = field(default=True)
        unc_label_smoothing: float = field(default=0.0)
        ref_adapter_name: str = field(default="_ref")
        kl_decay: float = field(default=1.0)

    def __init__(self, *args, tokenizer=None, val_data=None, test_data=None, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSupervisedDataset(tokenizer),
        )

        self.val_data = val_data
        self.test_data = test_data

    def compute_unc_loss(self, model, inputs, outputs):
        query_inputs, query_token_vec = prepare_query(
            self.tokenizer, inputs, outputs, format=self.args.query_format
        )
        query_inputs = self.data_collator(query_inputs)

        if self.args.unc_normalize:
            query_outputs = model(**query_inputs)

            _, unc_y, unc_logits = extract_qa_exact(
                self.tokenizer, query_inputs, outputs=query_outputs
            )
            unc_y, unc_logits = (
                (unc_y.unsqueeze(-1) == query_token_vec).long().argmax(dim=-1),
                unc_logits[:, query_token_vec],
            )

            unc_loss = F.cross_entropy(
                unc_logits,
                unc_y.to(unc_logits.device),
                label_smoothing=self.args.unc_label_smoothing,
            )

            return unc_loss

        unc_loss = super().compute_loss(model, query_inputs)

        return unc_loss

    def compute_kl_loss(self, model, inputs, outputs):
        with torch.inference_mode():
            model.module.set_adapter(self.args.ref_adapter_name)
            ref_outputs = model.module(**inputs)
            model.module.set_adapter("default")

        labels = inputs.get("labels")[..., 1:]
        probs = outputs.logits[..., :-1, :].softmax(dim=-1)
        ref_probs = ref_outputs.logits[..., :-1, :].softmax(dim=-1)
        # mix_probs = (probs + ref_probs) / 2

        p = Categorical(probs=probs)
        p_ref = Categorical(probs=ref_probs)

        # p_mix = Categorical(probs=mix_probs)
        # raw_loss = (kl_divergence(p, p_mix) + kl_divergence(p_ref, p_mix)) / 2

        raw_loss = kl_divergence(p, p_ref)

        loss_mask = labels != IGNORE_LABEL
        loss = (raw_loss * loss_mask).sum(dim=-1).mean(dim=0)

        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        unc_loss = self.compute_unc_loss(model, inputs, outputs)
        kl_loss = self.compute_kl_loss(model, inputs, outputs)

        total_loss = unc_loss + self.args.kl_decay * kl_loss

        loss_metrics = {
            "lm_loss": loss.detach().item(),
            "unc_loss": unc_loss.detach().item(),
            "kl_loss": kl_loss.detach().item(),
        }

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(loss_metrics)

        return (total_loss, outputs) if return_outputs else total_loss

    @torch.inference_mode()
    def evaluate(self, *_, **__):
        # metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        metrics = {}

        val_metrics, test_metrics = evaluate_dataset(
            self.accelerator,
            self.model,
            self.tokenizer,
            None,
            train_data=False,
            seed=self.args.seed,
            val_data=self.val_data,
            test_data=self.test_data,
            prompt_style="choice",
        )

        if val_metrics is not None:
            val_metrics = {f"eval/{k}": v for k, v in val_metrics.items()}
            self.log(val_metrics)
            metrics.update(val_metrics)

        if test_metrics is not None:
            test_metrics = {f"test/{k}": v for k, v in test_metrics.items()}
            self.log(test_metrics)
            metrics.update(test_metrics)

        return metrics
