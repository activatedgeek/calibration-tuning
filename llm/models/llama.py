import os
from timm.models import register_model
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
)

## FIXME: no fp32 on 24GB GPU?


def create_tokenizer(size, cache_dir=None, **_):
    assert size in ["7b", "13b", "30b", "65b"]

    tokenizer = LlamaTokenizer.from_pretrained(
        f"openlm-research/open_llama_{size}",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
    )
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token    
    return tokenizer


def create_model(size, cache_dir=None, load_in_8bit=True, device_map=None, **_):
    return LlamaForCausalLM.from_pretrained(
        f"openlm-research/open_llama_{size}",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        load_in_8bit=load_in_8bit,
        device_map=device_map,
    )

@register_model
def open_llama_7b_tokenizer(**kwargs):
    return create_tokenizer("7b", **kwargs)

@register_model
def open_llama_7b(**kwargs):
    return create_model("7b", **kwargs)

@register_model
def open_llama_13b_tokenizer(**kwargs):
    return create_tokenizer("13b", **kwargs)

@register_model
def open_llama_13b(**kwargs):
    return create_model("13b", **kwargs)
