import os
from dataclasses import dataclass, field
from tqdm.auto import tqdm
import torch
from transformers.trainer import (
    Trainer,
    logger,
    unwrap_model,
    TRAINING_ARGS_NAME,
    TrainingArguments,
)

from ..datasets import DictCollator, LabeledStringDataCollator


class FineTuner(Trainer):
    TEMPERATURE_WEIGHTS_NAME = "temperature_head.bin"

    @dataclass
    class Args(TrainingArguments):
        scale_temp: bool = field(default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            data_collator=DictCollator(),
        )

        self._collate_fn = LabeledStringDataCollator(self.tokenizer)

    def compute_loss(self, model, inputs, **kwargs):
        inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
        targets = [inp.pop("target") for inp in inputs]

        loss_inputs = {
            k: v.to(self.accelerator.device)
            for k, v in self._collate_fn(
                [{**inp, "target": t} for inp, t in zip(inputs, targets)]
            ).items()
        }

        return super().compute_loss(model, loss_inputs, **kwargs)

    def evaluate(self, eval_dataset=None, metric_key_prefix="eval", **_):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        all_metrics = {"loss": []}

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
                loss = super().compute_loss(
                    self.model, loss_inputs, return_outputs=False
                )

            ## De-mean for distributed only.
            loss = (
                torch.zeros(B)
                .index_fill_(0, torch.tensor([0]).long(), loss * B)
                .to(loss.device)
            )
            [
                all_metrics[l].append(v)
                for l, v in zip(
                    ("loss",),
                    self.accelerator.gather_for_metrics((loss,)),
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
                unwrap_model(self.model).lm_head[-1].state_dict(),
                os.path.join(output_dir, self.TEMPERATURE_WEIGHTS_NAME),
            )
