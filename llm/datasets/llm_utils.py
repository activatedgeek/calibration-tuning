from dataclasses import dataclass
import torch
import transformers


## NOTE: HF Convention. See https://huggingface.co/docs/transformers/v4.30.0/en/tasks/token_classification#preprocess.
IGNORE_LABEL = -100


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple(
            [torch.tensor(instance[key]) for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_LABEL
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def tokenize_for_causal_lm(tokenizer, sample):
    tokenize_dict = tokenizer(
        sample["source"] + sample["target"],
        padding="longest",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    labels = torch.tensor(tokenize_dict["input_ids"])
    source_len = (
        labels.eq(tokenizer.eos_token_id)
        .nonzero()[labels.eq(tokenizer.eos_token_id).sum(dim=-1).cumsum(dim=0) - 1]
        .item()
        - 1
    )
    labels[:source_len] = IGNORE_LABEL
    tokenize_dict["labels"] = labels.tolist()

    return tokenize_dict
