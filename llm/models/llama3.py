from peft import prepare_model_for_kbit_training
import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast, 
    LlamaForCausalLM,
)

from .registry import register_model
from .llm_model_utils import DEFAULT_PAD_TOKEN, resize_token_embeddings


def create_tokenizer(
    kind,
    model_dir=None,
    padding_side="left",
    model_max_length=None,
    **kwargs,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir or f"meta-llama/Meta-Llama-3-{kind}",
        padding_side=padding_side,
        model_max_length=8192,
        use_fast=True,
        legacy=False,
        use_auth_token=True,
        **kwargs,
    )

    tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})

    return tokenizer


def create_model(
    kind,
    torch_dtype=None,
    model_dir=None,
    use_cache=False,
    tokenizer=None,
    use_int8=False,
    use_int4=False,
    **kwargs,
):
    quantization_config = None
    if use_int8 or use_int4:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=use_int4,
            load_in_8bit=use_int8,
        )

    model = LlamaForCausalLM.from_pretrained(
        model_dir or f"meta-llama/Meta-Llama-3-{kind}",
        torch_dtype=torch_dtype
        or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
        quantization_config=quantization_config,
        use_cache=use_cache,
        **kwargs,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    resize_token_embeddings(tokenizer, model)

    if use_int8:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    return model


def create_tokenizer_and_model(kind, tokenizer_args=None, **kwargs):
    tokenizer = create_tokenizer(kind, **(tokenizer_args or dict()))
    model = create_model(kind, tokenizer=tokenizer, **kwargs)
    return tokenizer, model


@register_model
def llama3_8b_tokenizer(**kwargs):
    return create_tokenizer("8B", **kwargs)


@register_model
def llama3_8b(**kwargs):
    return create_tokenizer_and_model("8B", **kwargs)


@register_model
def llama3_8b_instruct_tokenizer(**kwargs):
    return create_tokenizer("8B-Instruct", **kwargs)


@register_model
def llama3_8b_instruct(**kwargs):
    return create_tokenizer_and_model("8B-Instruct", **kwargs)


@register_model
def llama3_70b_tokenizer(**kwargs):
    return create_tokenizer("70B", **kwargs)


@register_model
def llama3_70b(**kwargs):
    return create_tokenizer_and_model("70B", **kwargs)


@register_model
def llama3_70b_instruct_tokenizer(**kwargs):
    return create_tokenizer("70B-Instruct", **kwargs)


@register_model
def llama3_70b_instruct(**kwargs):
    return create_tokenizer_and_model("70B-Instruct", **kwargs)
