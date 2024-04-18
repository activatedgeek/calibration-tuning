from peft import prepare_model_for_kbit_training
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, MistralForCausalLM


from .registry import register_model
from .llm_model_utils import DEFAULT_PAD_TOKEN, resize_token_embeddings


__all__ = ["create_tokenizer", "create_model"]


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
def mistral_7b_tokenizer(**kwargs):
    return create_tokenizer("Mistral-7B-v0.1", **kwargs)


@register_model
def mistral_7b(**kwargs):
    return create_tokenizer_and_model("Mistral-7B-v0.1", **kwargs)


@register_model
def mistral_7b_instruct_tokenizer(**kwargs):
    return create_tokenizer("Mistral-7B-Instruct-v0.2", **kwargs)


@register_model
def mistral_7b_instruct(**kwargs):
    return create_tokenizer_and_model("Mistral-7B-Instruct-v0.2", **kwargs)


@register_model
def mixtral_8x22b_tokenizer(**kwargs):
    return create_tokenizer("Mixtral-8x22B-v0.1", **kwargs)


@register_model
def mixtral_8x22b(**kwargs):
    return create_tokenizer_and_model("Mixtral-8x22B-v0.1", **kwargs)


@register_model
def mixtral_8x22b_instruct_tokenizer(**kwargs):
    return create_tokenizer("Mixtral-8x22B-Instruct-v0.1", **kwargs)


@register_model
def mixtral_8x22b_instruct(**kwargs):
    return create_tokenizer_and_model("Mixtral-8x22B-Instruct-v0.1", **kwargs)
