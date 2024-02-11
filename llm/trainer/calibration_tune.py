import os
from dataclasses import dataclass, field
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
from transformers.trainer import (
    logger,
    unwrap_model,
    TRAINING_ARGS_NAME,
    Trainer,
    TrainingArguments,
)

from ..datasets import (
    IGNORE_LABEL,
    DictCollator,
    LabeledStringDataCollator,
    prepare_uncertainty_query,
)


class CalibrationTuner(Trainer):
    TEMPERATURE_WEIGHTS_NAME = "query_temperature_head.bin"

    @dataclass
    class Args(TrainingArguments):
        use_lm_loss: bool = field(default=False)
        query_format: str = field(default="roman_choice")
        ref_adapter_name: str = field(default="_ref")
        unc_label_smoothing: float = field(default=0.0)
        kl_type: str = field(default="jsd")
        kl_decay: float = field(default=0.0)
        scale_temp: bool = field(default=False)

    def __init__(self, *args, query_temperature_model=None, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            data_collator=DictCollator(),
        )

        self._collate_fn = LabeledStringDataCollator(self.tokenizer)
        self.query_temperature_model = query_temperature_model

    def _wrap_model(self, *args, **kwargs):
        if self.args.scale_temp:
            if (
                unwrap_model(self.query_temperature_model)
                is self.query_temperature_model
            ):
                self.query_temperature_model = self.accelerator.prepare(
                    self.query_temperature_model
                )

        return super()._wrap_model(*args, **kwargs)

    def compute_kl_loss(self, model, inputs, outputs):
        with torch.inference_mode():
            unwrapped_model = unwrap_model(model)
            unwrapped_model.set_adapter(self.args.ref_adapter_name)
            ref_outputs = unwrapped_model(**inputs)
            unwrapped_model.set_adapter("default")

            del unwrapped_model

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
            predictions = self.tokenizer.batch_decode(
                outputs.logits[:, -1, :].argmax(dim=-1),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            q_labels = None

        q_inputs, q_labels, q_token_vec = prepare_uncertainty_query(
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

        if self.args.scale_temp:
            q_logits = self.query_temperature_model(q_logits)

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
            loss = loss.detach()

        q_loss = self.compute_query_loss(
            model,
            inputs,
            targets,
            outputs,
        )

        kl_loss = self.compute_kl_loss(model, loss_inputs, outputs)

        loss_metrics = {
            "lm_loss": loss.detach().item(),
            "q_loss": q_loss.detach().item(),
            "kl_loss": kl_loss.detach().item(),
        }

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(loss_metrics)

        total_loss = loss + q_loss + self.args.kl_decay * kl_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def evaluate(self, eval_dataset=None, metric_key_prefix="eval", **_):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        all_metrics = {"loss": [], "q_loss": [], "kl_loss": []}

        for inputs in tqdm(eval_dataloader, leave=False):
            inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
            targets = [inp.pop("target") for inp in inputs]
            B = len(inputs)

            loss_inputs = {
                k: v.to(self.accelerator.device)
                for k, v in self._collate_fn(
                    [{**inp, "target": t} for inp, t in zip(inputs, targets)]
                ).items()
            }

            with torch.inference_mode():
                loss, outputs = super().compute_loss(
                    self.model, loss_inputs, return_outputs=True
                )

            if not self.args.use_lm_loss:
                loss = loss.detach()

            q_loss = self.compute_query_loss(
                self.model,
                inputs,
                targets,
                outputs,
            )

            kl_loss = self.compute_kl_loss(self.model, loss_inputs, outputs)

            ## De-mean for distributed only.
            loss = (
                torch.zeros(B)
                .index_fill_(0, torch.tensor([0]).long(), loss * B)
                .to(loss.device)
            )
            q_loss = (
                torch.zeros(B)
                .index_fill_(0, torch.tensor([0]).long(), q_loss * B)
                .to(q_loss.device)
            )
            kl_loss = (
                torch.zeros(B)
                .index_fill_(0, torch.tensor([0]).long(), kl_loss * B)
                .to(kl_loss.device)
            )
            [
                all_metrics[l].append(v)
                for l, v in zip(
                    ("loss", "q_loss", "kl_loss"),
                    self.accelerator.gather_for_metrics((loss, q_loss, kl_loss)),
                )
            ]

        all_metrics = {k: torch.cat(v, dim=0) for k, v in all_metrics.items()}
        N = all_metrics["loss"].size(0)

        all_metrics = {
            f"{metric_key_prefix}_{k}": v[v.nonzero().squeeze(-1)].sum() / N
            for k, v in all_metrics.items()
        }

        self.log(all_metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, all_metrics
        )

        return all_metrics

    def _save(self, output_dir=None, state_dict=None):
        ## NOTE: Fix for name hierarchy due to multiple adapters.
        if state_dict is None:
            state_dict = self.model.state_dict()
            state_dict.update(
                {".".join(k.split(".")[2:]): v for k, v in state_dict.items()}
            )

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
            selected_adapters=["default"],
            save_embedding_layers=False,
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        if self.args.scale_temp:
            torch.save(
                unwrap_model(self.query_temperature_model).state_dict(),
                os.path.join(output_dir, self.TEMPERATURE_WEIGHTS_NAME),
            )
