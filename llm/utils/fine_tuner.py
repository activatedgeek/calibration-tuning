import os
from dataclasses import dataclass, field
import torch
from transformers import Trainer
from transformers.trainer import logger, TRAINING_ARGS_NAME
from transformers.training_args import TrainingArguments

from ..datasets.llm_utils import DataCollatorForSupervisedDataset
from ..eval import evaluate_dataset
from ..models.peft import save_temperature_scaled_model


class FineTuner(Trainer):
    @dataclass
    class Args(TrainingArguments):
        scale_temp: bool = field(default=False)

    def __init__(self, *args, tokenizer=None, val_data=None, test_data=None, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSupervisedDataset(tokenizer),
        )

        self.val_data = val_data
        self.test_data = test_data

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
