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

import torch.nn as nn
from safetensors.torch import load_file as safe_load_file
from huggingface_hub import file_exists as hf_file_exists, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from peft.utils.other import infer_device

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
    use_temperature: bool = field(
        default=False, metadata={"help": "Temperature-scaled query probabilities."}
    )


class PeftModelForCalibratedCausalLM(PeftModelForCausalLM):
    TEMPERATURE_WEIGHTS_NAME = "temperature_model.pt"
    SAFETENSORS_TEMPERATURE_WEIGHTS_NAME = "temperature_model.safetensors"

    class TemperatureScale(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_temperature = nn.Parameter(torch.tensor(0.0))

        def forward(self, inputs):
            return inputs / self.log_temperature.exp()

    def _get_token_vec(self, tokenizer):
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

    def _prepare_uncertainty_query(self, contexts, predictions):
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

    def generate(
        self, *args, tokenizer=None, collate_fn=None, use_temperature=False, **kwargs
    ):
        use_temperature = use_temperature or self.active_peft_config.use_temperature

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

        q_token_vec = self._get_token_vec(tokenizer)
        q_str_inputs = self._prepare_uncertainty_query(str_inputs, str_outputs)

        q_str_inputs = [{"context": s} for s in q_str_inputs]
        q_inputs = {
            k: v.cuda(input_ids.device) for k, v in collate_fn(q_str_inputs).items()
        }

        q_logits = self(**q_inputs).logits[..., -1, q_token_vec]
        if use_temperature:
            q_logits = self.temperature_scale[self.active_adapter](q_logits)

        p_correct = q_logits.softmax(dim=-1)[:, 1]

        return outputs, p_correct

    def load_temperature_adapter(self, model_id, device=None, **hf_hub_download_kwargs):
        path = (
            os.path.join(model_id, hf_hub_download_kwargs["subfolder"])
            if hf_hub_download_kwargs.get("subfolder", None) is not None
            else model_id
        )

        if device is None:
            device = infer_device()

        if os.path.exists(
            os.path.join(path, self.SAFETENSORS_TEMPERATURE_WEIGHTS_NAME)
        ):
            filename = os.path.join(path, self.SAFETENSORS_TEMPERATURE_WEIGHTS_NAME)
            use_safetensors = True
        elif os.path.exists(os.path.join(path, self.TEMPERATURE_WEIGHTS_NAME)):
            filename = os.path.join(path, self.TEMPERATURE_WEIGHTS_NAME)
            use_safetensors = False
        else:
            token = hf_hub_download_kwargs.get("token", None)
            if token is None:
                token = hf_hub_download_kwargs.get("use_auth_token", None)

            hub_filename = (
                os.path.join(
                    hf_hub_download_kwargs["subfolder"],
                    self.SAFETENSORS_TEMPERATURE_WEIGHTS_NAME,
                )
                if hf_hub_download_kwargs.get("subfolder", None) is not None
                else self.SAFETENSORS_TEMPERATURE_WEIGHTS_NAME
            )
            has_remote_safetensors_file = hf_file_exists(
                repo_id=model_id,
                filename=hub_filename,
                revision=hf_hub_download_kwargs.get("revision", None),
                repo_type=hf_hub_download_kwargs.get("repo_type", None),
                token=token,
            )

            use_safetensors = has_remote_safetensors_file

            if has_remote_safetensors_file:
                filename = hf_hub_download(
                    model_id,
                    self.SAFETENSORS_TEMPERATURE_WEIGHTS_NAME,
                    **hf_hub_download_kwargs,
                )
            else:
                try:
                    filename = hf_hub_download(
                        model_id,
                        self.TEMPERATURE_WEIGHTS_NAME,
                        **hf_hub_download_kwargs,
                    )
                except EntryNotFoundError:
                    filename = None

        if filename:
            if use_safetensors:
                if hasattr(torch.backends, "mps") and (device == torch.device("mps")):
                    temperature_weights = safe_load_file(filename, device="cpu")
                else:
                    temperature_weights = safe_load_file(filename, device=device)
            else:
                temperature_weights = torch.load(
                    filename, map_location=torch.device(device)
                )

            return temperature_weights

    def load_adapter(self, model_id, adapter_name, **kwargs):
        load_result = super().load_adapter(model_id, adapter_name, **kwargs)

        hf_hub_download_kwargs, _ = self._split_kwargs(kwargs)

        temperature_weights = self.load_temperature_adapter(
            model_id, **hf_hub_download_kwargs
        )

        if not hasattr(self, "temperature_scale"):
            self.temperature_scale = dict()

        self.temperature_scale[adapter_name] = self.TemperatureScale()
        if temperature_weights:
            self.temperature_scale[adapter_name].load_state_dict(temperature_weights)

        return load_result


## Hot patch config/model mapping.
PEFT_TYPE_TO_CONFIG_MAPPING["CALIBRATED_LORA"] = CalibratedLoraConfig
MODEL_TYPE_TO_PEFT_MODEL_MAPPING["CALIBRATED_CAUSAL_LM"] = (
    PeftModelForCalibratedCausalLM
)


@torch.inference_mode
def main(model_name=None, max_new_tokens=100, output_path='/workspace/output_ambiguity.csv', use_temp=False):

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
                    # cache_dir=os.environ.get("HF_MODELS_CACHE"),
                )
                model.peft_config["query"].use_temperature = use_temp
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
