import os
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
)

from .registry import register_model


__all__ = ["create_tokenizer", "create_model"]


def create_tokenizer(
    size=None, model_dir=None, cache_dir=None, padding_side="right", use_fast=False, **_
):
    if size is not None:
        assert size in ["7b"]

    tokenizer = LlamaTokenizer.from_pretrained(
        model_dir or f"huggyllama/llama-{size}",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        padding_side=padding_side,
        use_fast=use_fast,
    )
    return tokenizer


def create_model(size, model_dir=None, cache_dir=None, **kwargs):
    if size is not None:
        assert size in ["7b"]

    return LlamaForCausalLM.from_pretrained(
        model_dir or f"huggyllama/llama-{size}",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        **kwargs,
    )


@register_model
def llama_7b_tokenizer(**kwargs):
    return create_tokenizer("7b", **kwargs)


@register_model
def llama_7b(**kwargs):
    return create_model("7b", **kwargs)
