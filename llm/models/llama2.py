import logging
from peft import prepare_model_for_kbit_training
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from ..datasets import LabeledStringDataCollator
from .registry import register_model
from .llm_model_utils import DEFAULT_PAD_TOKEN, resize_token_embeddings


__HF_MODEL_MAP = {
    "7b": "Llama-2-7b-hf",
    "7b-chat": "Llama-2-7b-chat-hf",
    "13b": "Llama-2-13b-hf",
    "13b-chat": "Llama-2-13b-chat-hf",
    "70b": "Llama-2-70b-hf",
    "70b-chat": "Llama-2-70b-chat-hf",
}


def __get_model_hf_id(model_str):
    try:
        _, kind = model_str.split(":")

        assert kind in __HF_MODEL_MAP.keys()
    except ValueError:
        logging.exception(
            f'Model string should be formatted as "llama2:<kind>" (Got {model_str})',
        )
        raise
    except AssertionError:
        logging.exception(
            f'Model not found. Model string should be formatted as "llama2:<kind>" (Got {model_str})',
        )
        raise

    return __HF_MODEL_MAP[kind]


def create_tokenizer(
    kind,
    model_dir=None,
    padding_side="left",
    model_max_length=4096,
    **kwargs,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir or f"meta-llama/{kind}",
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
    if use_int8 or use_int4:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=use_int4,
            load_in_8bit=use_int8,
        )

    model = LlamaForCausalLM.from_pretrained(
        model_dir or f"meta-llama/{kind}",
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


class LMEmbedModel:
    def __init__(self, t, m):
        self.tokenizer = t
        self.model = m
        self.tokenizer_args = LabeledStringDataCollator.get_tokenizer_args(
            self.tokenizer
        )

    @torch.inference_mode
    def __call__(self, texts):
        inputs = self.tokenizer(texts, **self.tokenizer_args)
        inputs.pop("length", None)

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1][..., -1, :]

        return embeddings.clone()


def create_embed_model(kind, **kwargs):
    return LMEmbedModel(*create_tokenizer_and_model(kind, **kwargs))


@register_model
def llama2_tokenizer(*, model_str=None, **kwargs):
    return create_tokenizer(__get_model_hf_id(model_str), **kwargs)


@register_model
def llama2(*, model_str=None, **kwargs):
    return create_tokenizer_and_model(__get_model_hf_id(model_str), **kwargs)


@register_model
def llama2_embed(*, model_str=None, **kwargs):
    return create_embed_model(__get_model_hf_id(model_str), **kwargs)
