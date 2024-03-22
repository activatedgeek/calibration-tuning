import os
import json
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

from llm.datasets import LabeledStringDataCollator, LMText
from llm.models import get_model

import os
import wandb
import pandas as pd
from tqdm.auto import tqdm
from transformers import GenerationConfig

# import multiprocess.context as ctx

# ## @HOTFIX: for hanging processes in dataset map.
# ctx._force_start_method("spawn")

from llm.datasets.llm_utils_oe import grade_oe_preds #prepare_uncertainty_query
from llm.logging import entrypoint_with_accelerator
from llm.models import get_model
from llm.models.peft import get_lora_model
from llm.utils.generate_utils import generate_output


def get_query_labels(
    inputs,
    targets,
    predictions,
    strategy="substring",
    format="roman_choice",
):
    # contexts = [str(LMText.from_(inp)) for inp in inputs]

    query_labels = grade_oe_preds(
        targets, predictions, inputs, strategy, mode="answer-key"
    )

    return torch.Tensor(query_labels).long()


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
def main(model_name=None, max_new_tokens=100, output_path='/workspace/output_ambiguity.csv'):

    queries = {}
    outputs = {}
    P = {}
    response = {}
    correctness = {}

    # for adapter_name in 'default', 'query':
    for adapter_name in ['query']:

        with open('SelfAware.json') as f:
            self_aware = json.load(f)

            tokenizer, model = get_model(model_name, device_map="cuda")

            if adapter_name == 'query':
                model = PeftModel.from_pretrained(
                    model,
                    f"calibration-tuning/{model.config._name_or_path.split('/')[-1]}-ct-oe",
                    adapter_name="query",
                    cache_dir=os.environ.get("HF_MODELS_CACHE"),
                )
            else:
                pass
                # keep at base.

            model.eval()

            generation_config = GenerationConfig(
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            collate_fn = LabeledStringDataCollator(tokenizer)

            for ex in [
                ex for ex in self_aware['example'] if not ex['answerable']
            ]:
                query = ex['question']
                target = "The question is not answerable; I do not know."

                inputs = {k: v.cuda() for k, v in collate_fn([{"context": query}]).items()}

                id = ex['question_id']
                if id not in queries.keys():
                    outputs[id] = {}
                    P[id] = {}
                    response[id] = {}
                    correctness[id] = {}
                    queries[id] = query

                outputs[id][adapter_name], P[id][adapter_name] = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    tokenizer=tokenizer,
                    collate_fn=collate_fn,
                )

                response[id][adapter_name] = tokenizer.batch_decode(
                    outputs[id][adapter_name][:, inputs.get("input_ids").size(-1) :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                correctness[id][adapter_name] = get_query_labels(
                    [query],
                    [target],
                    response[id][adapter_name],
                    format="roman_choice",
                    strategy="fuzzy_gpt-3.5-turbo-1106",
                )

                # import pdb; pdb.set_trace()

            # print(f"(Pinocchio says with {P[0] * 100:.1f}% confidence)> {response[0]}")
            # print()


        # csv_path = f"{log_dir}/labels/{split}"
        # with accelerator.main_process_first():
        #     if accelerator.is_main_process:
        #         os.makedirs(csv_path)

        # pd.DataFrame(label_generator).to_csv(
        #     f"{csv_path}/{accelerator.process_index}.csv", index=False
        # )

        pd.DataFrame(label_generator).to_csv(
            output_path, index=False
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
