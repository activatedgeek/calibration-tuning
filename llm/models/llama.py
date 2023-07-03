import os
import torch
from timm.models import register_model
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
)


def create_tokenizer(size, cache_dir=None, padding_side="left", **_):
    assert size in ["7b", "13b", "30b", "65b"]

    tokenizer = LlamaTokenizer.from_pretrained(
        f"openlm-research/open_llama_{size}",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        padding_side=padding_side,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


## NOTE: no fp32 on 24GB GPU?
def create_model(size, cache_dir=None, torch_dtype=torch.float16, **kwargs):
    assert size in ["7b", "13b", "30b", "65b"]

    kwargs = { k: v for k, v in kwargs.items() if not k.startswith("pretrained")}

    return LlamaForCausalLM.from_pretrained(
        f"openlm-research/open_llama_{size}",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        torch_dtype=torch_dtype,
        **kwargs,
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
