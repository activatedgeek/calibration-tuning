import os
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
from transformers import Trainer
from transformers.trainer import logger, TRAINING_ARGS_NAME
from transformers.training_args import TrainingArguments

from ..datasets import IGNORE_LABEL, DictCollator, LabeledStringDataCollator
from ..datasets.llm_utils_oe import prepare_oe_uncertainty_query
from ..eval import evaluate_dataset
from ..models.peft import save_temperature_scaled_model


class UncertaintyTuner(Trainer):
    @dataclass
    class Args(TrainingArguments):
        use_lm_loss: bool = field(default=False)
        query_format: str = field(default="roman_choice")
        ref_adapter_name: str = field(default="_ref")
        unc_label_smoothing: float = field(default=0.0)
        kl_type: str = field(default="jsd")
        kl_decay: float = field(default=0.0)
        scale_temp: bool = field(default=False)

    def __init__(self, *args, tokenizer=None, val_data=None, test_data=None, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            tokenizer=tokenizer,
            data_collator=DictCollator(),
        )

        self._collate_fn = LabeledStringDataCollator(tokenizer)

        self.val_data = val_data
        self.test_data = test_data

    def compute_kl_loss(self, model, inputs, outputs):
        with torch.inference_mode():
            model.module.set_adapter(self.args.ref_adapter_name)
            ref_outputs = model.module(**inputs)
            model.module.set_adapter("default")

        labels = inputs.get("labels")[..., 1:]
        probs = outputs.logits[..., :-1, :].softmax(dim=-1)
        ref_probs = ref_outputs.logits[..., :-1, :].softmax(dim=-1)

        p = Categorical(probs=probs)
        p_ref = Categorical(probs=ref_probs)

        if self.args.kl_type == "reverse_kl":
            kl_loss = kl_divergence(p, p_ref)
        elif self.args.kl_type == "forward_kl":
            kl_loss = kl_divergence(p_ref, p)
        elif self.args.kl_type == "jsd":
            p_mix = Categorical(probs=(probs + ref_probs) / 2)
            kl_loss = (kl_divergence(p, p_mix) + kl_divergence(p_ref, p_mix)) / 2
        else:
            raise NotImplementedError

        loss_mask = labels != IGNORE_LABEL
        loss = (kl_loss * loss_mask).sum(dim=-1).mean(dim=0)

        return loss

    def compute_query_loss(self, model, inputs, targets, outputs):
        if "query_label" in inputs[0]:
            predictions = [inp.pop("output") for inp in inputs]
            q_labels = [inp.pop("query_label") for inp in inputs]
        else:
            predictions = outputs.logits[:, -1, :].argmax(dim=-1)
            ## TODO: handle query label construction for mcq.
            raise NotImplementedError

        q_inputs, q_labels, q_token_vec = prepare_oe_uncertainty_query(
            self.tokenizer,
            inputs,
            targets,
            predictions,
            query_labels=q_labels,
            format=self.args.query_format,
        )

        q_generation_inputs = {
            k: v.to(self.accelerator.device)
            for k, v in self._collate_fn(q_inputs).items()
        }

        q_generation_outputs = model(**q_generation_inputs)
        q_logits = q_generation_outputs.logits[..., -1, q_token_vec]

        q_loss = F.cross_entropy(
            q_logits,
            q_labels.to(q_logits.device),
            label_smoothing=self.args.unc_label_smoothing,
        )

        return q_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
        targets = [inp.pop("target") for inp in inputs]

        loss_inputs = {
            k: v.to(self.accelerator.device)
            for k, v in self._collate_fn(
                [{**inp, "target": t} for inp, t in zip(inputs, targets)]
            ).items()
        }

        loss, outputs = super().compute_loss(model, loss_inputs, return_outputs=True)
        if not self.args.use_lm_loss:
            loss = torch.zeros_like(loss).detach()

        q_loss = self.compute_query_loss(
            model,
            inputs,
            targets,
            outputs,
        )

        kl_loss = (
            torch.tensor(0.0)
            if self.args.scale_temp
            else self.compute_kl_loss(model, loss_inputs, outputs)
        )

        total_loss = loss + q_loss + self.args.kl_decay * kl_loss

        loss_metrics = {
            "lm_loss": loss.detach().item(),
            "q_loss": q_loss.detach().item(),
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

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
            selected_adapters=["default"],
        )

        if self.args.scale_temp:
            save_temperature_scaled_model(self.model, output_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
