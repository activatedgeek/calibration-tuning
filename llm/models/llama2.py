import os
from timm.models import register_model
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
)


__all__ = ["create_tokenizer", "create_model"]


def create_tokenizer(
    size=None, model_dir=None, cache_dir=None, padding_side="right", use_fast=False, **_
):
    if size is not None:
        assert size in ["7b", "13b", "7b-chat", "13b-chat"]

    tokenizer = LlamaTokenizer.from_pretrained(
        model_dir or f"meta-llama/Llama-2-{size}",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        padding_side=padding_side,
        use_fast=use_fast,
        use_auth_token=True,
    )
    return tokenizer


def create_model(size=None, model_dir=None, cache_dir=None, **kwargs):
    if size is not None:
        assert size in ["7b", "13b", "7b-chat", "13b-chat"]

    kwargs = {k: v for k, v in kwargs.items() if not k.startswith("pretrained")}

    return LlamaForCausalLM.from_pretrained(
        model_dir or f"meta-llama/Llama-2-{size}",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        use_auth_token=True,
        **kwargs,
    )


@register_model
def llama2_7b_tokenizer(**kwargs):
    return create_tokenizer("7b", **kwargs)


@register_model
def llama2_7b(**kwargs):
    return create_model("7b", **kwargs)


@register_model
def llama2_13b_tokenizer(**kwargs):
    return create_tokenizer("13b", **kwargs)


@register_model
def llama2_13b(**kwargs):
    return create_model("13b", **kwargs)
