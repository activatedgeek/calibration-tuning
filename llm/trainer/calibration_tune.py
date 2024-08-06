import os
from dataclasses import dataclass, field
from tqdm.auto import tqdm
import torch
from torch.distributions import Categorical, kl_divergence
import torch.nn.functional as F
from torch.utils.data import default_collate
from transformers.trainer import (
    logger,
    TRAINING_ARGS_NAME,
    Trainer,
    TrainingArguments,
)

from ..datasets import (
    IGNORE_LABEL,
    LabeledStringDataCollator,
    prepare_uncertainty_query,
)


class CalibrationTuner(Trainer):
    TEMPERATURE_WEIGHTS_NAME = "query_temperature_head.bin"

    @dataclass
    class Args(TrainingArguments):
        fp16: bool = field(default=not torch.cuda.is_bf16_supported())
        bf16: bool = field(default=torch.cuda.is_bf16_supported())
        ddp_find_unused_parameters: bool = field(default=False)
        log_on_each_node: bool = field(default=False)
        eval_strategy: str = field(default="steps")
        dataloader_num_workers: int = field(default=4)
        optim: str = field(default="adamw_torch")
        lr: float = field(default=1e-4)
        lr_scheduler_type: str = field(default="cosine")
        weight_decay: float = field(default=0.0)
        warmup_ratio: float = field(default=0.0)
        gradient_accumulation_steps: int = field(default=1)
        report_to: str = field(default="wandb")
        ## Custom Args.
        use_lm_loss: bool = field(default=False)
        query_format: str = field(default="roman_choice")
        ref_adapter_name: str = field(default="_ref")
        unc_label_smoothing: float = field(default=0.0)
        kl_type: str = field(default="jsd")
        kl_decay: float = field(default=0.0)
        scale_temp: bool = field(default=False)

    def __init__(
        self,
        args=None,
        train_dataset=None,
        tokenizer=None,
        query_temperature_model=None,
        **kwargs,
    ):
        args.label_names = train_dataset.column_names

        self._collate_fn = LabeledStringDataCollator(tokenizer)

        self.query_temperature_model = query_temperature_model

        super().__init__(
            **kwargs,
            args=args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            data_collator=default_collate,
        )

    def _wrap_model(self, *args, **kwargs):
        if self.args.scale_temp:
            if (
                self.accelerator.unwrap_model(self.query_temperature_model)
                is self.query_temperature_model
            ):
                self.query_temperature_model = self.accelerator.prepare(
                    self.query_temperature_model
                )

        return super()._wrap_model(*args, **kwargs)

    def compute_lm_loss(self, model, inputs, targets):
        if "query_label" in inputs[0]:
            predictions = [inp.pop("output") for inp in inputs]
            lm_loss = torch.tensor(0.0)
        else:
            generation_inputs = {
                k: v.to(self.accelerator.device)
                for k, v in self._collate_fn(
                    [{**inp, "target": t} for inp, t in zip(inputs, targets)]
                ).items()
            }

            with torch.inference_mode(mode=not self.args.use_lm_loss):
                lm_loss, generation_outputs = model(
                    **generation_inputs, return_outputs=True
                )

            predictions = self.tokenizer.batch_decode(
                generation_outputs.logits[:, -1, :].argmax(dim=-1),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        assert (
            lm_loss.requires_grad == self.args.use_lm_loss
        ), f"Expected lm_loss to be detached."

        return predictions, lm_loss

    def compute_query_loss(self, model, inputs, targets, predictions):
        q_labels = (
            [inp.pop("query_label") for inp in inputs]
            if "query_label" in inputs[0]
            else None
        )

        q_inputs, q_labels, q_token_vec = prepare_uncertainty_query(
            self.tokenizer,
            inputs,
            targets,
            predictions,
            query_labels=q_labels,
            format=self.args.query_format,
        )

        q_loss_inputs = {
            k: v.to(self.accelerator.device)
            for k, v in self._collate_fn(q_inputs).items()
        }

        q_outputs = model(**q_loss_inputs)
        q_logits = q_outputs.logits[..., -1, q_token_vec]

        if self.args.scale_temp:
            q_logits = self.query_temperature_model(q_logits)

        q_loss = F.cross_entropy(
            q_logits,
            q_labels.to(q_logits.device),
            label_smoothing=self.args.unc_label_smoothing,
        )

        return q_loss

    def compute_kl_loss(self, model, inputs, targets):
        if self.args.kl_decay <= 0.0:
            return torch.tensor(0.0)

        ref_inputs = {
            k: v.to(self.accelerator.device)
            for k, v in self._collate_fn(
                [{**inp, "target": t} for inp, t in zip(inputs, targets)]
            ).items()
        }

        probs = model(**ref_inputs).logits[..., :-1, :].softmax(dim=-1)

        with torch.inference_mode():
            ## NOTE: self.model is always unwrapped.
            self.model.set_adapter(self.args.ref_adapter_name)

            ref_probs = self.model(**ref_inputs).logits[..., :-1, :].softmax(dim=-1)

            self.model.set_adapter("default")

        labels = ref_inputs.pop("labels")[..., 1:]

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

    def compute_loss(self, model, inputs, return_outputs=False, return_metrics=False):
        inputs.pop("embedding", None)
        inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
        targets = [inp.pop("target") for inp in inputs]

        predictions, lm_loss = self.compute_lm_loss(model, inputs, targets)

        q_loss = self.compute_query_loss(
            model,
            inputs,
            targets,
            predictions,
        )

        kl_loss = self.compute_kl_loss(model, inputs, targets)

        loss_metrics = {
            "lm_loss": lm_loss.detach().item(),
            "q_loss": q_loss.detach().item(),
            "kl_loss": kl_loss.detach().item(),
        }

        if return_metrics:
            return loss_metrics

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(loss_metrics)

        loss = lm_loss + q_loss + self.args.kl_decay * kl_loss

        return (loss, None) if return_outputs else loss

    def evaluate(self, eval_dataset=None, metric_key_prefix="eval", **_):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        all_metrics = {"lm_loss": [], "q_loss": [], "kl_loss": []}

        for inputs in tqdm(eval_dataloader, leave=False):
            B = len(inputs.get("target"))

            with torch.inference_mode():
                loss_metrics = self.compute_loss(
                    self.model_wrapped, inputs, return_metrics=True
                )

            ## De-mean for distributed computation. Size for correct gather.
            loss_metrics = {
                k: torch.zeros(B)
                .index_fill_(0, torch.tensor([0]).long(), v * B)
                .to(self.accelerator.device)
                for k, v in loss_metrics.items()
            }

            [
                all_metrics[l].append(v)
                for l, v in zip(
                    all_metrics.keys(),
                    self.accelerator.gather_for_metrics(
                        tuple(loss_metrics[k] for k in all_metrics.keys())
                    ),
                )
            ]

        all_metrics = {k: torch.cat(v, dim=0) for k, v in all_metrics.items()}
        N = all_metrics["q_loss"].size(0)

        all_metrics = {
            f"{metric_key_prefix}_{k}": (v[v.nonzero().squeeze(-1)].sum() / N).item()
            for k, v in all_metrics.items()
        }
        all_metrics[f"{metric_key_prefix}_N"] = N

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
                self.accelerator.unwrap_model(
                    self.query_temperature_model
                ).state_dict(),
                os.path.join(output_dir, self.TEMPERATURE_WEIGHTS_NAME),
            )
