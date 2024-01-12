import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .utils import get_last_checkpoint_path


WEIGHTS_NAME = "temperature_adapter.bin"


class TemperatureScaledLlamaForCausalLM(LlamaForCausalLM):
    PARAMETER_NAME = "log_temperature"

    def __init__(self, *args, temp_scaling=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.temp_scaling = temp_scaling

        self.register_parameter(
            self.PARAMETER_NAME,
            nn.Parameter(torch.zeros(1), requires_grad=False),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        scale_temp=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        ########## BEGIN MODIFICATION ##########

        should_scale = scale_temp if scale_temp is not None else self.temp_scaling

        if should_scale:
            T = getattr(self, self.PARAMETER_NAME).exp()
            logits = logits / T

        ########## END MODIFICATION ##########

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def prepare_model_for_temperature_scaling(model, is_trainable=False, peft_dir=None):
    if peft_dir is not None:
        peft_dir = get_last_checkpoint_path(peft_dir)

    for n, p in model.named_parameters():
        if TemperatureScaledLlamaForCausalLM.PARAMETER_NAME in n:
            p.requires_grad_(is_trainable)
            p.data = torch.zeros_like(p.data)

            if peft_dir is not None and os.path.isfile(f"{peft_dir}/{WEIGHTS_NAME}"):
                ckpt_params = torch.load(
                    f"{peft_dir}/{WEIGHTS_NAME}", map_location=p.data.device
                )
                ## Only one value in checkpoint file.
                p.data = list(ckpt_params.values())[0]

                logging.info(f"Temperature loaded from {peft_dir}.")
        else:
            if p.requires_grad:
                p.requires_grad_(False)

    logging.info(f"Model prepared for temperature scaling.")


def save_temperature_scaled_model(model, output_dir):
    ## Assumes scaling used only once, strip of module path for independent reloading.
    state_dict = {
        k: v
        for k, v in model.state_dict().items()
        if TemperatureScaledLlamaForCausalLM.PARAMETER_NAME in k
    }

    if not len(state_dict):
        logging.warning(
            f"Parameter {TemperatureScaledLlamaForCausalLM.PARAMETER_NAME} not found. Skipping save."
        )
        return

    with open(f"{output_dir}/{WEIGHTS_NAME}", "wb") as f:
        torch.save(state_dict, f)

    logging.debug(state_dict)
