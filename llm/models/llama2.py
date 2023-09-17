import os
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
)

from .registry import register_model
from .llm_utils import get_special_tokens


__all__ = ["create_tokenizer", "create_model"]


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
    if causal_lm:
        model = LlamaForCausalLM.from_pretrained(
            model_dir or f"meta-llama/Llama-2-{size}-hf",
            cache_dir=os.environ.get("MODELDIR", cache_dir),
            **kwargs,
        )
    else:
        model = LlamaForSequenceClassification.from_pretrained(
            model_dir or f"meta-llama/Llama-2-{size}-hf",
            cache_dir=os.environ.get("MODELDIR", cache_dir),
            **kwargs,
        )

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
