from dataclasses import dataclass, asdict as dataclassasdict
import torch
import transformers
from datasets.formatting.formatting import LazyRow


## NOTE: HF Convention. See https://huggingface.co/docs/transformers/en/tasks/token_classification#preprocess.
IGNORE_LABEL = -100


@dataclass
class StringDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        return self.tokenizer(
            [str(LMText.from_(instance)) for instance in instances],
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
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


def tokenize_for_causal_lm(tokenizer, sample, prompt_style="choice"):
    ## NOTE: Hope that no truncation by model length is needed, or else the logic fails.
    tokenizer_args = dict(
        padding="longest",
        truncation=True,
        # max_length=tokenizer.model_max_length,
    )

    sample = LMText.from_(sample)

    tokenize_dict = tokenizer(str(sample), **tokenizer_args)

    labels = torch.tensor(tokenize_dict["input_ids"]) if sample.target else None

    if prompt_style == "choice":
        if labels is not None:
            ## Target is 1 token length only.
            source_len = (
                labels.eq(tokenizer.eos_token_id)
                .nonzero()[
                    labels.eq(tokenizer.eos_token_id).sum(dim=-1).cumsum(dim=0) - 1
                ]
                .item()
                - 1
            )
    elif prompt_style == "oe":
        if labels is not None:
            ## Encoded answer, except first BOS token id.
            target_ids = torch.tensor(
                tokenizer(sample.target.strip(), **tokenizer_args)["input_ids"]
            )[1:]
            source_len = len(labels) - len(target_ids)

            # assert torch.allclose(labels[source_len:], target_ids)
    else:
        raise NotImplementedError

    if labels is not None:
        labels[:source_len] = IGNORE_LABEL
        tokenize_dict["labels"] = labels.tolist()

    if sample.query_label is not None:
        tokenize_dict["query_label"] = sample.query_label

        ## NOTE: Skip BOS token at start during tokenization of outputs.
        output_tokenize_dict = tokenizer(sample.output, **tokenizer_args)
        tokenize_dict["output_ids"] = output_tokenize_dict["input_ids"][1:]

    return tokenize_dict


def tokenize_datasets(tokenizer, *datasets, num_workers=8, **kwargs):
    return [
        data.map(
            lambda x: tokenize_for_causal_lm(tokenizer, x, **kwargs),
            num_proc=num_workers,
            # remove_columns=list(LMText.__annotations__.keys()),
        )
        if data is not None
        else None
        for data in datasets
    ]


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


def prepare_batch(tokenizer, inputs, **kwargs):
    """
    Assumes dictionary inputs with item values as lists.
    """
    if all([isinstance(v, torch.Tensor) for v in inputs.values()]):
        return inputs

    return [
        tokenize_for_causal_lm(
            tokenizer,
            dict(zip(inputs.keys(), vals)),
            **kwargs,
        )
        for vals in zip(*inputs.values())
    ]


def prepare_query(
    tokenizer,
    inputs,
    outputs,
    offline_outputs=None,
    offline_query_labels=None,
    format="roman_choice",
):
    if offline_query_labels is None:
        eos_idx, y, logits = extract_qa_exact(tokenizer, inputs, outputs=outputs)
        y_hat = logits.argmax(dim=-1)

        response_ids = inputs.get("input_ids").clone()
        response_ids[torch.arange(response_ids.size(0)), eos_idx] = y_hat

        ## Remove the initial BOS token to conflict with BOS added during tokenization.
        assert (
            response_ids[:, 0] == tokenizer.bos_token_id
        ).sum() == response_ids.size(
            0
        ), "Not all sequences start with BOS token. This should not happen."
        response_ids = response_ids[:, 1:]

        responses = tokenizer.batch_decode(response_ids)
        query_labels = y == y_hat
    else:
        ## argmax gets first mismatch, i.e. where labels start.
        ctx_lengths = (
            (inputs.get("input_ids") == inputs.get("labels")).long().argmax(dim=-1)
        )

        responses = [
            tokenizer.decode(
                inp[:ctx_len],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            + tokenizer.decode(
                out,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for inp, out, ctx_len in zip(
                inputs.get("input_ids"), offline_outputs, ctx_lengths
            )
        ]
        query_labels = offline_query_labels

    if format == "bool":
        ## NOTE: Probably don't use, often seems to be biased towards a yes.
        query_inputs = [
            tokenize_for_causal_lm(
                tokenizer,
                {
                    "context": f"{r}\n\nIs the proposed answer correct? ",
                    "target": ("yes" if a else "no") + tokenizer.eos_token,
                },
                prompt_style="choice",
            )
            for r, a in zip(responses, query_labels)
        ]
    elif format == "alpha_choice":
        query_inputs = [
            tokenize_for_causal_lm(
                tokenizer,
                {
                    "context": f"{r}\n\nIs the proposed answer correct?\n\nChoices:\n(a): no\n(b): yes",
                    "target_prompt": "\nAnswer:",
                    "target": ("b" if a else "a") + tokenizer.eos_token,
                },
                prompt_style="choice",
            )
            for r, a in zip(responses, query_labels)
        ]
    elif format == "roman_choice":
        query_inputs = [
            tokenize_for_causal_lm(
                tokenizer,
                {
                    "context": f"{r}\n\nIs the proposed answer correct?\n\nChoices:\n(i): no\n(ii): yes",
                    "target_prompt": "\nAnswer:",
                    "target": ("ii" if a else "i") + tokenizer.eos_token,
                },
                prompt_style="choice",
            )
            for r, a in zip(responses, query_labels)
        ]
    else:
        raise NotImplementedError

    return query_inputs, get_token_vec(tokenizer, format=format)
