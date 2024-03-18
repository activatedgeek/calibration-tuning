import os

import torch
from transformers import BitsAndBytesConfig, LlamaTokenizer, LlamaForCausalLM

from .registry import register_model
from .llm_model_utils import DEFAULT_PAD_TOKEN, resize_token_embeddings


def create_tokenizer(
    kind,
    model_dir=None,
    cache_dir=None,
    padding_side="left",
    model_max_length=None,
    **kwargs,
):
    tokenizer = LlamaTokenizer.from_pretrained(
        model_dir or f"meta-llama/Llama-2-{kind}-hf",
        cache_dir=os.environ.get("HF_MODELS_CACHE", cache_dir),
        padding_side=padding_side,
        model_max_length=model_max_length,
        use_fast=True,
        legacy=False,
        **kwargs,
    )

    tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})

    return tokenizer


def create_model(
    kind,
    torch_dtype=None,
    model_dir=None,
    cache_dir=None,
    use_cache=False,
    tokenizer=None,
    use_int8=False,
    **kwargs,
):
    model = LlamaForCausalLM.from_pretrained(
        model_dir or f"meta-llama/Llama-2-{kind}-hf",
        torch_dtype=torch_dtype
        or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
        quantization_config=(
            BitsAndBytesConfig(
                load_in_8bit=True,
            )
            if use_int8
            else None
        ),
        cache_dir=os.environ.get("HF_MODELS_CACHE", cache_dir),
        use_cache=use_cache,
        **kwargs,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    resize_token_embeddings(tokenizer, model)

    return model


def create_tokenizer_and_model(kind, tokenizer_args=None, **kwargs):
    tokenizer = create_tokenizer(kind, **(tokenizer_args or dict()))
    model = create_model(kind, tokenizer=tokenizer, **kwargs)
    return tokenizer, model


@register_model
def llama2_7b_tokenizer(**kwargs):
    return create_tokenizer("7b", **kwargs)


@register_model
def llama2_7b(**kwargs):
    return create_tokenizer_and_model("7b", **kwargs)


@register_model
def llama2_7b_chat_tokenizer(**kwargs):
    return create_tokenizer("7b-chat", **kwargs)


@register_model
def llama2_7b_chat(**kwargs):
    return create_tokenizer_and_model("7b-chat", **kwargs)


@register_model
def llama2_13b_tokenizer(**kwargs):
    return create_tokenizer("13b", **kwargs)


@register_model
def llama2_13b(**kwargs):
    return create_tokenizer_and_model("13b", **kwargs)


@register_model
def llama2_13b_chat_tokenizer(**kwargs):
    return create_tokenizer("13b-chat", **kwargs)


@register_model
def llama2_13b_chat(**kwargs):
    return create_tokenizer_and_model("13b-chat", **kwargs)


@register_model
def llama2_70b_tokenizer(**kwargs):
    return create_tokenizer("70b", **kwargs)


@register_model
def llama2_70b(**kwargs):
    return create_tokenizer_and_model("70b", **kwargs)


@register_model
def llama2_70b_chat_tokenizer(**kwargs):
    return create_tokenizer("70b-chat", **kwargs)


@register_model
def llama2_70b_chat(**kwargs):
    return create_tokenizer_and_model("70b-chat", **kwargs)
