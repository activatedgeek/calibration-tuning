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
        sample["source"] + sample["target"], padding=True, truncation=True
    )

    source_len = (
        torch.Tensor(
            tokenizer(sample["source"], padding=True, truncation=True).input_ids
        )
        .long()
        .ne(tokenizer.pad_token_id)
        .sum()
        .item()
    )

    labels = torch.Tensor(tokenize_dict["input_ids"]).long()
    labels[:source_len] = IGNORE_LABEL
    tokenize_dict["labels"] = labels.tolist()

    return tokenize_dict
