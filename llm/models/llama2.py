import os

from transformers import LlamaTokenizer, LlamaForCausalLM

from .registry import register_model
from .llm_utils import get_special_tokens, resize_token_embeddings


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
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def create_model(
    size=None,
    model_dir=None,
    cache_dir=None,
    tokenizer=None,
    causal_lm=True,
    **kwargs,
):
    model = LlamaForCausalLM.from_pretrained(
        model_dir or f"meta-llama/Llama-2-{size}-hf",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        **kwargs,
    )

    resize_token_embeddings(tokenizer, model, causal_lm=causal_lm)

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
