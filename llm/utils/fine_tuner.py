from dataclasses import dataclass
from transformers import Trainer
from transformers.training_args import TrainingArguments

from ..datasets import DictCollator, LabeledStringDataCollator


class FineTuner(Trainer):
    @dataclass
    class Args(TrainingArguments): ...

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

    ## Skip eval.
    def evaluate(self, *_, **__):
        # metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        metrics = {}
        return metrics
