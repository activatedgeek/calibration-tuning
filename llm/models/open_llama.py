import os
from timm.models import register_model
from transformers import (
    LlamaTokenizer,
    LlamaForSequenceClassification,
    LlamaForCausalLM,
)


def __filter_kwargs(kwargs):
    """Filter extraneous keys not used in HF"""
    return {k: v for k, v in kwargs.items() if not k.startswith("pretrained")}


## NOTE: no fp32 on 24GB GPU?
@register_model
def open_llama_13b(cache_dir=None, load_in_8bit=True, device_map=None, **_):
    # kwargs = __filter_kwargs(kwargs)
    return LlamaForCausalLM.from_pretrained(
        "openlm-research/open_llama_13b",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        load_in_8bit=load_in_8bit,
        device_map=device_map,
    )


@register_model
def open_llama_13b_tokenizer(cache_dir=None, **kwargs):
    kwargs = __filter_kwargs(kwargs)
    tokenizer = LlamaTokenizer.from_pretrained(
        "openlm-research/open_llama_13b",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
