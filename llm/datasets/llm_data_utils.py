from enum import Enum
from dataclasses import dataclass, asdict as dataclassasdict
import torch
import transformers
from datasets.formatting.formatting import LazyRow


## NOTE: HF Convention. See https://huggingface.co/docs/transformers/en/tasks/token_classification#preprocess.
IGNORE_LABEL = -100


class PromptFormat(str, Enum):
    CHOICE = "choice"
    OE = "oe"


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

LLAMA_3_SYS_PROMPT = "You are an expert who responds with concise, correct answers. Directly state the answer without phrases like 'the correct answer is'"

@dataclass
class LabeledStringDataCollator:
    tokenizer: transformers.PreTrainedTokenizer
    target_name: str = "target"

    @staticmethod
    def get_tokenizer_args(tokenizer):
        return dict(
            padding=True,
            truncation=True,
            max_length=(
                tokenizer.model_max_length
                if hasattr(tokenizer, "model_max_length")
                else None
            ),
            return_tensors="pt",
            return_length=True,
        )

    def __call__(self, instances):
        tokenizer_args = self.get_tokenizer_args(self.tokenizer)

        prompts = [str(LMText.from_(instance)) for instance in instances]

        if self.tokenizer.name_or_path and \
            ('Llama-3' in self.tokenizer.name_or_path) and \
            ('Instruct' in self.tokenizer.name_or_path):
            msgs = [
                [
                    {"role": "system", "content": LLAMA_3_SYS_PROMPT},
                    {"role": "user", "content": p}
                ] for p in prompts
            ]

            prompts = [
                self.tokenizer.apply_chat_template(
                    m, 
                    tokenize=False, 
                    add_generation_prompt=True
                ) for m in msgs
            ]
            
        inputs = self.tokenizer(prompts, **tokenizer_args)
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
