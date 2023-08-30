import os
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
)

from .registry import register_model


__all__ = ["create_tokenizer", "create_model"]


def create_tokenizer(
    size=None, model_dir=None, cache_dir=None, padding_side="right", use_fast=False, **_
):
    tokenizer = LlamaTokenizer.from_pretrained(
        model_dir or f"meta-llama/Llama-2-{size}-hf",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        padding_side=padding_side,
        use_fast=use_fast,
        legacy=False,
    )
    return tokenizer


def create_model(size=None, model_dir=None, cache_dir=None, causal_lm=True, **kwargs):
    if causal_lm:
        return LlamaForCausalLM.from_pretrained(
            model_dir or f"meta-llama/Llama-2-{size}-hf",
            cache_dir=os.environ.get("MODELDIR", cache_dir),
            **kwargs,
        )
    else:
        return LlamaForSequenceClassification.from_pretrained(
            model_dir or f"meta-llama/Llama-2-{size}-hf",
            cache_dir=os.environ.get("MODELDIR", cache_dir),
            **kwargs,
        )

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