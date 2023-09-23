import os
import copy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from peft import PeftModel

from transformers import (
    LlamaTokenizer,
    LlamaModel,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaPreTrainedModel
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.utils import WEIGHTS_NAME

from .registry import register_model
from .llm_utils import get_special_tokens


__all__ = ["create_tokenizer", "create_model"]


class LlamaForSequenceClassificationFFNN(LlamaPreTrainedModel):
    def __init__(self, base_model, **kwargs):
        super().__init__(base_model.config)
        self.model = base_model
        self.base_model_is_peft = isinstance(base_model, PeftModel)

        self.num_labels = 2
        self.score = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_labels, bias=False),
        )
        
        for module in self.score.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight) 

        self.score.to(self.model.device)       

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.base_model_is_peft:
            output_hidden_states = True

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.base_model_is_peft:
            hidden_states = transformer_outputs.hidden_states[-1].float()
        else:
            hidden_states = transformer_outputs[0]

        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def save_pretrained(self, save_directory, **kwargs):
        state_dict = copy.deepcopy(self.state_dict())

        for k in list(state_dict.keys()):
            if "model" in k:
                del state_dict[k]

        torch.save(state_dict, os.path.join(save_directory, WEIGHTS_NAME))

    @classmethod
    def from_pretrained(cls, base_model, model_path, device_map=None, **kwargs):
        model = cls(base_model)

        state_dict = torch.load(os.path.join(model_path, WEIGHTS_NAME))
        model.load_state_dict(state_dict, strict=False)
        model.score.to(model.model.device)

        return model

def create_tokenizer(size=None, model_dir=None, cache_dir=None, **_):
    tokenizer = LlamaTokenizer.from_pretrained(
        model_dir or f"meta-llama/Llama-2-{size}-hf",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        padding_side="left",
        use_fast=False,
        legacy=False,
    )

    tokenizer.add_special_tokens(get_special_tokens(tokenizer))

    return tokenizer


def create_model(
    size=None, model_dir=None, cache_dir=None, causal_lm=True, tokenizer=None, **kwargs
):
    base_model = kwargs.pop('base_model', None)
    if causal_lm:
        model = LlamaForCausalLM.from_pretrained(
            model_dir or f"meta-llama/Llama-2-{size}-hf",
            cache_dir=os.environ.get("MODELDIR", cache_dir),
            **kwargs,
        )
    else:        
        if model_dir is None:
            model = LlamaForSequenceClassificationFFNN(base_model, **kwargs)
        else:
            model = LlamaForSequenceClassificationFFNN.from_pretrained(
                base_model,
                model_dir,
                cache_dir=os.environ.get("MODELDIR", cache_dir),
                **kwargs,
            )

        return model

    extra_token_count = len(tokenizer) - model.get_input_embeddings().weight.data.size(
        0
    )
    if extra_token_count:
        model.resize_token_embeddings(len(tokenizer))

        input_embeddings = model.get_input_embeddings().weight.data

        input_embeddings[-extra_token_count:] = input_embeddings[
            :-extra_token_count
        ].mean(dim=0, keepdim=True)

        if causal_lm:
            output_embeddings = model.get_output_embeddings().weight.data

            output_embeddings[-extra_token_count:] = output_embeddings[
                :-extra_token_count
            ].mean(dim=0, keepdim=True)

    return model


@register_model
def llama2_7b_tokenizer(**kwargs):
    return create_tokenizer("7b", **kwargs)


@register_model
def llama2_7b(**kwargs):
    return create_model("7b", **kwargs)


@register_model
def llama2_7b_chat_tokenizer(**kwargs):
    return create_tokenizer("7b-chat", **kwargs)


@register_model
def llama2_7b_chat(**kwargs):
    return create_model("7b-chat", **kwargs)


@register_model
def llama2_13b_tokenizer(**kwargs):
    return create_tokenizer("13b", **kwargs)


@register_model
def llama2_13b(**kwargs):
    return create_model("13b", **kwargs)


@register_model
def llama2_13b_chat_tokenizer(**kwargs):
    return create_tokenizer("13b-chat", **kwargs)


@register_model
def llama2_13b_chat(**kwargs):
    return create_model("13b-chat", **kwargs)


@register_model
def llama2_70b_tokenizer(**kwargs):
    return create_tokenizer("70b", **kwargs)


@register_model
def llama2_70b(**kwargs):
    return create_model("70b", **kwargs)


@register_model
def llama2_70b_chat_tokenizer(**kwargs):
    return create_tokenizer("70b-chat", **kwargs)


@register_model
def llama2_70b_chat(**kwargs):
    return create_model("70b-chat", **kwargs)
