import os
import fire
from dataclasses import dataclass, field
import torch

from transformers import GenerationConfig
from peft import (
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
    PEFT_TYPE_TO_CONFIG_MAPPING,
    PeftModel,
    LoraConfig,
    PeftModelForCausalLM,
)

from llm.datasets import LabeledStringDataCollator
from llm.models import get_model


@dataclass
class CalibratedLoraConfig(LoraConfig):
    query_format: int = field(
        default="roman_choice", metadata={"help": "Query format."}
    )


class PeftModelForCalibratedCausalLM(PeftModelForCausalLM):
    def get_token_vec(self, tokenizer):
        query_format = self.active_peft_config.query_format

        vocab = tokenizer.get_vocab()

        def _create_vec(raw_list):
            for t in raw_list:
                assert t in vocab, f"Cannot handle {t} as a single token."

            return torch.tensor([tokenizer(t).input_ids[-1] for t in raw_list])

        if query_format == "roman_choice":
            raw_strings = ["i", "ii"]
        else:
            raise NotImplementedError(f'Format "{self.format}" not supported.')

        return _create_vec(raw_strings)

    def prepare_uncertainty_query(self, contexts, predictions):
        query_format = self.active_peft_config.query_format

        def _format_query_text(c, p):
            if query_format == "roman_choice":
                query_text = "\n".join(
                    [
                        c + p,
                        "\nIs the proposed answer correct?",
                        "Choices:",
                        "(i): no",
                        "(ii): yes",
                        "Answer:",
                    ]
                )
            else:
                raise NotImplementedError(f'Format "{query_format}" not supported.')

            return query_text

        query_inputs = [_format_query_text(c, p) for c, p in zip(contexts, predictions)]

        return query_inputs

    def generate(self, *args, tokenizer=None, collate_fn=None, **kwargs):
        with self.disable_adapter():
            outputs = super().generate(*args, **kwargs)

        input_ids = kwargs.get("input_ids")

        str_inputs = tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        str_outputs = tokenizer.batch_decode(
            outputs[:, input_ids.size(-1) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        q_token_vec = self.get_token_vec(tokenizer)
        q_str_inputs = self.prepare_uncertainty_query(str_inputs, str_outputs)

        q_str_inputs = [{"context": s} for s in q_str_inputs]
        q_inputs = {
            k: v.cuda(input_ids.device) for k, v in collate_fn(q_str_inputs).items()
        }

        q_outputs = self(**q_inputs)
        q_logits = q_outputs.logits[..., -1, q_token_vec].softmax(dim=-1)

        p_correct = q_logits[:, 1]

        return outputs, p_correct


## Hot patch config/model mapping.
PEFT_TYPE_TO_CONFIG_MAPPING["CALIBRATED_LORA"] = CalibratedLoraConfig
MODEL_TYPE_TO_PEFT_MODEL_MAPPING["CALIBRATED_CAUSAL_LM"] = (
    PeftModelForCalibratedCausalLM
)


@torch.inference_mode
def main(model_name=None, max_new_tokens=100):
    tokenizer, model = get_model(model_name, device_map="auto")

    model = PeftModel.from_pretrained(
        model,
        f"calibration-tuning/{model.config._name_or_path.split('/')[-1]}-ct-oe",
        adapter_name="query",
        cache_dir=os.environ.get("HF_MODELS_CACHE"),
    )
    model.eval()

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    collate_fn = LabeledStringDataCollator(tokenizer)

    while True:
        query = input("(Enter query)> ")

        inputs = {k: v.cuda() for k, v in collate_fn([{"context": query}]).items()}

        outputs, P = model.generate(
            **inputs,
            generation_config=generation_config,
            tokenizer=tokenizer,
            collate_fn=collate_fn,
        )

        response = tokenizer.batch_decode(
            outputs[:, inputs.get("input_ids").size(-1) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        print(f"(Pinocchio says with {P[0] * 100:.1f}% confidence)> {response[0]}")
        print()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
