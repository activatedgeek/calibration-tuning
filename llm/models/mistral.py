import logging
from peft import prepare_model_for_kbit_training
import torch
from transformers import AutoTokenizer, MistralForCausalLM


from .registry import register_model
from .llm_model_utils import DEFAULT_PAD_TOKEN, resize_token_embeddings


__MISTRAL_HF_MODEL_MAP = {
    "7b": "Mistral-7B-v0.1",
    "7b-instruct": "Mistral-7B-Instruct-v0.2",
}

__MIXTRAL_HF_MODEL_MAP = {
    "8x22b": "Mixtral-8x22B-v0.1",
    "8x22b-instruct": "Mixtral-8x22B-Instruct-v0.1",
}


def __get_model_hf_id(model_str, model_map):
    try:
        model_name, kind = model_str.split(":")

        assert kind in model_map.keys()
    except ValueError:
        logging.exception(
            f'Model string should be formatted as "{model_name}:<kind>" (Got {model_str})',
        )
        raise
    except AssertionError:
        logging.exception(
            f'Model not found. Model string should be formatted as "{model_name}:<kind>" (Got {model_str})',
        )
        raise

    return model_map[kind]


def create_tokenizer(
    kind,
    model_dir=None,
    padding_side="left",
    model_max_length=8192,
    **kwargs,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir or f"mistralai/{kind}",
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
    use_cache=False,
    tokenizer=None,
    use_int8=False,
    use_int4=False,
    **kwargs,
):
    quantization_config = None
    if use_int4 or use_int8:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=use_int4,
            load_in_8bit=use_int8,
        )

    model = MistralForCausalLM.from_pretrained(
        model_dir or f"mistralai/{kind}",
        torch_dtype=torch_dtype
        or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
        quantization_config=quantization_config,
        use_cache=use_cache,
        **kwargs,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    resize_token_embeddings(tokenizer, model)

    if use_int4 or use_int8:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    return model


def create_tokenizer_and_model(kind, tokenizer_args=None, **kwargs):
    tokenizer = create_tokenizer(kind, **(tokenizer_args or dict()))
    model = create_model(kind, tokenizer=tokenizer, **kwargs)
    return tokenizer, model


@register_model
def mistral_tokenizer(*, model_str=None, **kwargs):
    return create_tokenizer(
        __get_model_hf_id(model_str, __MISTRAL_HF_MODEL_MAP), **kwargs
    )


@register_model
def mistral(*, model_str=None, **kwargs):
    return create_tokenizer_and_model(
        __get_model_hf_id(model_str, __MISTRAL_HF_MODEL_MAP), **kwargs
    )


@register_model
def mixtral_tokenizer(*, model_str=None, **kwargs):
    return create_tokenizer(
        __get_model_hf_id(model_str, __MIXTRAL_HF_MODEL_MAP), **kwargs
    )


@register_model
def mixtral(*, model_str=None, **kwargs):
    return create_tokenizer_and_model(
        __get_model_hf_id(model_str, __MIXTRAL_HF_MODEL_MAP), **kwargs
    )
