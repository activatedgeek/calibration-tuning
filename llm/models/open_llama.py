import os
from timm.models import register_model
from transformers import (
    LlamaForSequenceClassification,
    LlamaForCausalLM,
)


def __filter_kwargs(kwargs):
    """Filter extraneous keys not used in HF"""
    return {k: v for k, v in kwargs.items() if not k.startswith("pretrained")}


@register_model
def seq_open_llama_13b(cache_dir=None, num_classes=None, **kwargs):
    kwargs = __filter_kwargs(kwargs)
    return LlamaForSequenceClassification.from_pretrained(
        "openlm-research/open_llama_13b",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        num_labels=num_classes,
        **kwargs
    )


@register_model
def open_llama_13b(cache_dir=None, **kwargs):
    kwargs = __filter_kwargs(kwargs)
    return LlamaForCausalLM.from_pretrained(
        "openlm-research/open_llama_13b",
        cache_dir=os.environ.get("MODELDIR", cache_dir),
        **kwargs
    )
