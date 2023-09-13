import string
from dataclasses import dataclass
import torch
import transformers


## NOTE: HF Convention. See https://huggingface.co/docs/transformers/en/tasks/token_classification#preprocess.
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


def tokenize_datasets(tokenizer, *datasets, num_workers=8):
    return [
        data.map(
            lambda x: tokenize_for_causal_lm(tokenizer, x),
            num_proc=num_workers,
            remove_columns=["source", "target"],
        )
        if data is not None
        else None
        for data in datasets
    ]


def get_uq_answer_token_vec(tokenizer):
    _no_token = tokenizer("no").input_ids
    _yes_token = tokenizer("yes").input_ids

    ## NOTE: Assumes yes/no are single token (length 2 including BOS token).
    assert len(_no_token) == 2, f'Cannot handle "no" token {_no_token} yet.'
    assert len(_yes_token) == 2, f'Cannot handle "yes" token {_yes_token} yet.'

    return torch.tensor([_no_token[-1], _yes_token[-1]])


def get_options_token_vec(tokenizer, n=4):
    options = [tokenizer(o).input_ids for o in string.ascii_lowercase[:n]]

    for o, t in zip(options, string.ascii_lowercase[:n]):
        assert len(o) == 2, f'Cannot handle "{t}" token, found {o}'

    return torch.tensor([o[-1] for o in options])


def extract_qa_exact(tokenizer, inputs, outputs=None):
    """
    Assumes all answers are 1 token and end immediately with EOS token.
    """
    labels = inputs.get("labels")[..., 1:]

    eos_idx = labels.eq(tokenizer.eos_token_id).nonzero()[
        labels.eq(tokenizer.eos_token_id).sum(dim=-1).cumsum(dim=0) - 1
    ][:, -1]

    y = labels[torch.arange(labels.size(0)), eos_idx - 1]

    if outputs is not None:
        logits = outputs.logits[..., :-1, :]
        logits = logits[torch.arange(logits.size(0)), eos_idx - 1]

        return eos_idx, y, logits

    return eos_idx, y


def prepare_unc_query(tokenizer, inputs, outputs):
    eos_idx, y, logits = extract_qa_exact(tokenizer, inputs, outputs=outputs)
    y_hat = logits.argmax(dim=-1)

    response_ids = inputs.get("input_ids").clone()
    response_ids[torch.arange(response_ids.size(0)), eos_idx] = y_hat

    ## Remove the initial BOS token to conflict with BOS added during tokenization.
    assert (response_ids[:, 0] == tokenizer.bos_token_id).sum() == response_ids.size(
        0
    ), "Not all sequences start with BOS token. This should not happen."
    response_ids = response_ids[:, 1:]

    responses = tokenizer.batch_decode(response_ids)

    query_inputs = [
        tokenize_for_causal_lm(
            tokenizer,
            {
                "source": f"{r}\n\nIs the proposed answer correct?\n\nChoices:\n(a): no\n(b): yes\nAnswer: ",
                "target": string.ascii_lowercase[
                    ["no", "yes"].index("yes" if a else "no")
                ]
                + tokenizer.eos_token,
            },
        )
        for r, a in zip(responses, y == y_hat)
    ]

    return query_inputs
