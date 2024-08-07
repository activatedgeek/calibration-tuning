import os
from dataclasses import dataclass, field
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
from peft import PeftModel
from transformers.trainer import (
    TRAINING_ARGS_NAME,
    logger,
    Trainer,
    TrainingArguments,
)

from ..datasets import (
    LMText,
    LabeledStringDataCollator,
    prepare_uncertainty_query,
)


class EmbeddingTuner(Trainer):
    WEIGHTS_NAME = "classifier_model.bin"

    @dataclass
    class Args(TrainingArguments):
        accelerator_config: dict = field(
            default_factory=lambda: dict(use_configured_state=True)
        )
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

    def __init__(
        self,
        args=None,
        train_dataset=None,
        tokenizer=None,
        embedding_model=None,
        classifier_model=None,
        **kwargs,
    ):
        args.label_names = train_dataset.column_names

        self._collate_fn = LabeledStringDataCollator(tokenizer)

        self.embedding_model = embedding_model
        self.classifier_model = classifier_model

        super().__init__(
            **kwargs,
            args=args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            data_collator=default_collate,
        )

    def _wrap_model(self, *args, **kwargs):
        if (
            self.accelerator.unwrap_model(self.classifier_model)
            is self.classifier_model
        ):
            self.classifier_model = self.accelerator.prepare(self.classifier_model)

        return super()._wrap_model(*args, **kwargs)

    def prepare_inputs(self, model, inputs):
        embeddings = inputs.pop("embedding", None)
        inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
        targets = [inp.pop("target") for inp in inputs]

        if "query_label" in inputs[0]:
            predictions = [inp.pop("output") for inp in inputs]
            q_labels = torch.tensor([inp.pop("query_label") for inp in inputs]).long()
        else:
            generation_inputs = {
                k: v.to(self.accelerator.device)
                for k, v in self._collate_fn(inputs).items()
            }

            unwrapped_model = self.accelerator.unwrap_model(model)
            if isinstance(unwrapped_model, PeftModel):
                unwrapped_model.disable_adapter_layers()

            with torch.inference_mode():
                generation_outputs = unwrapped_model(
                    **generation_inputs, output_hidden_states=True
                )

            predictions = self.tokenizer.batch_decode(
                generation_outputs.logits[:, -1, :].argmax(dim=-1),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            _, q_labels, _ = prepare_uncertainty_query(
                self.tokenizer,
                inputs,
                targets,
                predictions,
                strategy="substring",
            )

            del generation_outputs

        q_labels = q_labels.to(self.accelerator.device)

        return inputs, targets, predictions, q_labels, embeddings

    def prepare_class_inputs(self, inputs, predictions, embeddings):
        if isinstance(embeddings, torch.Tensor):
            class_inputs = embeddings
        else:
            sentences = [
                str(LMText.from_({**i, "target": p}))
                for i, p in zip(inputs, predictions)
            ]

            class_inputs = self.embedding_model.encode(
                sentences, convert_to_tensor=True, show_progress_bar=False
            ).to(self.accelerator.device)

        return class_inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs, _, predictions, class_labels, embeddings = self.prepare_inputs(
            model, inputs
        )

        class_inputs = self.prepare_class_inputs(inputs, predictions, embeddings)

        class_logits = self.classifier_model(class_inputs)

        loss = F.cross_entropy(class_logits, class_labels)

        loss_metrics = {
            "loss": loss.detach().item(),
        }

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(loss_metrics)

        return (loss, None) if return_outputs else loss

    @torch.inference_mode
    def evaluate(self, eval_dataset=None, metric_key_prefix="eval", **_):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        all_labels, all_logits = [], []

        for inputs in tqdm(eval_dataloader, leave=False):
            inputs, _, predictions, class_labels, embeddings = self.prepare_inputs(
                self.model, inputs
            )

            class_inputs = self.prepare_class_inputs(inputs, predictions, embeddings)

            class_logits = self.classifier_model(class_inputs)

            [
                l.append(v)
                for l, v in zip(
                    (all_labels, all_logits),
                    self.accelerator.gather_for_metrics((class_labels, class_logits)),
                )
            ]

        all_labels = torch.cat(all_labels, dim=0)
        all_logits = torch.cat(all_logits, dim=0)

        metrics = {
            f"{metric_key_prefix}_N": all_labels.size(0),
            f"{metric_key_prefix}_acc": (all_logits.argmax(dim=-1) == all_labels)
            .float()
            .mean()
            .item(),
            f"{metric_key_prefix}_loss": F.cross_entropy(all_logits, all_labels).item(),
        }

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

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
            save_embedding_layers=False,
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        torch.save(
            self.accelerator.unwrap_model(self.classifier_model).state_dict(),
            os.path.join(output_dir, self.WEIGHTS_NAME),
        )
