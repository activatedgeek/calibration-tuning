from dataclasses import dataclass, asdict as dataclassasdict
import torch
import transformers
from datasets.formatting.formatting import LazyRow


## NOTE: HF Convention. See https://huggingface.co/docs/transformers/en/tasks/token_classification#preprocess.
IGNORE_LABEL = -100


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        raise ValueError("Does not use left padding, and will be erroneous.")

        input_ids = [torch.tensor(instance["input_ids"]) for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        batch_dict = dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "labels" in instances[0].keys():
            labels = [torch.tensor(instance["labels"]) for instance in instances]
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_LABEL
            )
            batch_dict["labels"] = labels

        if "query_label" in instances[0].keys():
            query_labels = torch.tensor(
                [instance["query_label"] for instance in instances]
            )
            batch_dict["query_label"] = query_labels

            output_ids = [
                torch.tensor(instance["output_ids"]) for instance in instances
            ]
            output_ids = torch.nn.utils.rnn.pad_sequence(
                output_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

            batch_dict["output_ids"] = output_ids

        return batch_dict


@dataclass
class LMText:
    context: str
    prompt: str = ""
    target_prompt: str = ""
    target: str = ""

    ## Misc.
    source_dataset: str = None
    output: str = None
    query_label: int = None

    def __str__(self):
        return (
            self.prompt + self.context + self.target_prompt + " " + self.target
        ).strip()

    def to_pydict(self):
        return {k: v for k, v in dataclassasdict(self).items() if v is not None}

    @staticmethod
    def from_(instance):
        if isinstance(instance, LMText):
            return instance

        if isinstance(instance, LazyRow):
            instance = {k: v for k, v in zip(instance.keys(), instance.values())}

        assert isinstance(
            instance, dict
        ), f"Could not convert instance to dict. Found {type(instance)}"

        return LMText(**instance)


def get_token_vec(tokenizer, format="roman_choice"):
    vocab = tokenizer.get_vocab()

    def _create_vec(raw_list):
        for t in raw_list:
            assert t in vocab, f"Cannot handle {t} as a single token."

        return torch.tensor([tokenizer(t).input_ids[-1] for t in raw_list])

    if format == "bool":
        raw_strings = ["no", "yes"]
    elif format == "alpha_choice":
        raw_strings = ["a", "b"]
    elif format == "mcq":
        raw_strings = ["a", "b", "c", "d"]
    elif format == "roman_choice":
        raw_strings = ["i", "ii"]
    else:
        raise NotImplementedError

    return _create_vec(raw_strings)
