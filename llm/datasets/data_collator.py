from dataclasses import dataclass
import transformers

from .llm_utils import LMText, IGNORE_LABEL


class DictCollator:
    def __call__(self, instances):
        return {k: [ins[k] for ins in instances] for k in instances[0].keys()}


@dataclass
class LabeledStringDataCollator:
    tokenizer: transformers.PreTrainedTokenizer
    target_name: str = "target"

    def __call__(self, instances):
        tokenizer_args = dict(
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            return_length=True,
        )

        inputs = self.tokenizer(
            [str(LMText.from_(instance)) for instance in instances],
            **tokenizer_args,
        )
        input_lengths = inputs.pop("length")

        if self.target_name in instances[0]:
            ## inputs without targets for labeling lengths.
            un_inputs = self.tokenizer(
                [
                    str(
                        LMText.from_(
                            {k: v for k, v in instance.items() if k != self.target_name}
                        )
                    )
                    for instance in instances
                ],
                **tokenizer_args,
            )
            un_input_lengths = un_inputs.pop("length")

            labels = inputs.get("input_ids").clone()
            for i, l in enumerate(input_lengths - un_input_lengths):
                labels[i, :-l] = IGNORE_LABEL
            inputs["labels"] = labels

        return inputs
