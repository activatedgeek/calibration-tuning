import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from .registry import register_model
from .llm_utils import get_special_tokens, resize_token_embeddings


__all__ = ["create_tokenizer", "create_model"]


def create_tokenizer(
    model_id=None, model_dir=None, cache_dir=None, model_max_length=8192, **_
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir or f"mistralai/{model_id}",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        padding_side="left",
        use_fast=True,
        legacy=False,
        model_max_length=model_max_length,
    )

    tokenizer.add_special_tokens(get_special_tokens(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def create_model(
    model_id=None,
    model_dir=None,
    cache_dir=None,
    tokenizer=None,
    **kwargs,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir or f"mistralai/{model_id}",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        **kwargs,
    )

    resize_token_embeddings(tokenizer, model)

    return model


@register_model
def mistral_7b_tokenizer(**kwargs):
    return create_tokenizer(**kwargs, model_id="Mistral-7B-v0.1")


@register_model
def mistral_7b(**kwargs):
    return create_model(**kwargs, model_id="Mistral-7B-v0.1")


@register_model
def mistral_7b_instruct_tokenizer(**kwargs):
    return create_tokenizer(**kwargs, model_id="Mistral-7B-Instruct-v0.2")


@register_model
def mistral_7b_instruct(**kwargs):
    return create_model(
        **kwargs,
        model_id="Mistral-7B-Instruct-v0.2",
    )
