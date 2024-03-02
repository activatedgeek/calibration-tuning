import os
from dataclasses import dataclass, field
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers.trainer import (
    TRAINING_ARGS_NAME,
    logger,
    unwrap_model,
    Trainer,
    TrainingArguments,
)

from ..datasets import (
    DictCollator,
    LabeledStringDataCollator,
    prepare_uncertainty_query,
)


class ClassificationTuner(Trainer):
    WEIGHTS_NAME = "classifier_model.bin"

    @dataclass
    class Args(TrainingArguments):
        target_layer: int = field(default=-1)

    def __init__(self, *args, classifier_model=None, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            data_collator=DictCollator(),
        )

        self._collate_fn = LabeledStringDataCollator(self.tokenizer)
        self.classifier_model = classifier_model

    def _wrap_model(self, *args, **kwargs):
        if unwrap_model(self.classifier_model) is self.classifier_model:
            self.classifier_model = self.accelerator.prepare(self.classifier_model)

        return super()._wrap_model(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        model = unwrap_model(model)

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

            if isinstance(model, PeftModel):
                model.set_adapter("default")

            with torch.inference_mode():
                generation_outputs = model(
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

        if isinstance(model, PeftModel) and "query" in model.peft_config:
            model.set_adapter("query")

            class_inputs, _, _ = prepare_uncertainty_query(
                self.tokenizer,
                inputs,
                targets,
                predictions,
                strategy="substring",
                query_labels=q_labels.cpu().numpy().tolist(),
            )
        else:
            model.set_adapter("default")

            class_inputs = [{**inp, "target": t} for inp, t in zip(inputs, predictions)]

        class_inputs = {
            k: v.to(self.accelerator.device)
            for k, v in self._collate_fn(class_inputs).items()
        }

        with torch.inference_mode():
            class_outputs = model(**class_inputs, output_hidden_states=True)

        class_inputs = class_outputs.hidden_states[self.args.target_layer][
            ..., -1, :
        ].clone()
        class_labels = q_labels.to(class_inputs.device)

        class_logits = self.classifier_model(class_inputs)

        loss = F.cross_entropy(class_logits, class_labels)

        loss_metrics = {
            "loss": loss.detach().item(),
        }

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(loss_metrics)

        return (loss, generation_outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, metric_key_prefix="eval", **_):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        all_labels, all_logits = [], []

        for inputs in tqdm(eval_dataloader, leave=False):
            inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
            targets = [inp.pop("target") for inp in inputs]

            if "query_label" in inputs[0]:
                predictions = [inp.pop("output") for inp in inputs]
                q_labels = torch.tensor(
                    [inp.pop("query_label") for inp in inputs]
                ).long()
            else:
                generation_inputs = {
                    k: v.to(self.accelerator.device)
                    for k, v in self._collate_fn(inputs).items()
                }

                with torch.inference_mode():
                    generation_outputs = self.model(
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

            class_inputs = {
                k: v.to(self.accelerator.device)
                for k, v in self._collate_fn(
                    [{**inp, "target": t} for inp, t in zip(inputs, predictions)]
                ).items()
            }

            if isinstance(self.model, PeftModel):
                self.model.set_adapter("default")

            with torch.inference_mode():
                class_outputs = self.model(**class_inputs, output_hidden_states=True)

            class_inputs = class_outputs.hidden_states[self.args.target_layer][
                ..., -1, :
            ].clone()
            class_labels = q_labels.to(class_inputs.device)

            class_logits = self.classifier_model(class_inputs.clone())

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
            unwrap_model(self.classifier_model).state_dict(),
            os.path.join(output_dir, self.WEIGHTS_NAME),
        )
