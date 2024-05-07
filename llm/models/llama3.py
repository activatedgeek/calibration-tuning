import logging

from .registry import register_model
from .llama2 import create_tokenizer, create_tokenizer_and_model, create_embed_model

TOKENIZER_ARGS = dict(model_max_length=8192)


__HF_MODEL_MAP = {
    "8b": "Meta-Llama-3-8B",
    "8b-instruct": "Meta-Llama-3-8B-Instruct",
    "70b": "Meta-Llama-3-70B",
    "70b-instruct": "Meta-Llama-3-70B-Instruct",
}


def __get_model_hf_id(model_str):
    try:
        _, kind = model_str.split(":")

        assert kind in __HF_MODEL_MAP.keys()
    except ValueError:
        logging.exception(
            f'Model string should be formatted as "llama3:<kind>" (Got {model_str})',
        )
        raise
    except AssertionError:
        logging.exception(
            f'Model not found. Model string should be formatted as "llama3:<kind>" (Got {model_str})',
        )
        raise

    return __HF_MODEL_MAP[kind]


@register_model(**TOKENIZER_ARGS)
def llama3_tokenizer(*, model_str=None, **kwargs):
    return create_tokenizer(__get_model_hf_id(model_str), **kwargs)


@register_model(tokenizer_args=TOKENIZER_ARGS)
def llama3(*, model_str=None, **kwargs):
    return create_tokenizer_and_model(__get_model_hf_id(model_str), **kwargs)


@register_model(tokenizer_args=TOKENIZER_ARGS)
def llama3_embed(*, model_str=None, **kwargs):
    return create_embed_model(__get_model_hf_id(model_str), **kwargs)
