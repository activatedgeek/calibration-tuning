import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from .registry import register_model
from .llm_utils import get_special_tokens


__all__ = ["create_tokenizer", "create_model"]


def create_tokenizer(cache_dir=None, **_):
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        padding_side="left",
        use_fast=True,
        legacy=False,
    )

    tokenizer.add_special_tokens(get_special_tokens(tokenizer))

    return tokenizer


def create_model(model_dir=None, cache_dir=None, tokenizer=None, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir or "mistralai/Mistral-7B-v0.1",
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

        output_embeddings = model.get_output_embeddings().weight.data

        output_embeddings[-extra_token_count:] = output_embeddings[
            :-extra_token_count
        ].mean(dim=0, keepdim=True)

    return model


@register_model
def mistral_7b_tokenizer(**kwargs):
    return create_tokenizer(**kwargs)


@register_model
def mistral_7b(**kwargs):
    return create_model(**kwargs)
