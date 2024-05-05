from .registry import register_model
from .llama2 import create_tokenizer, create_tokenizer_and_model, create_embed_model

TOKENIZER_ARGS = dict(model_max_length=8192)


@register_model(**TOKENIZER_ARGS)
def llama3_8b_tokenizer(**kwargs):
    return create_tokenizer("Meta-Llama-3-8B", **kwargs)


@register_model(tokenizer_args=TOKENIZER_ARGS)
def llama3_8b(**kwargs):
    return create_tokenizer_and_model("Meta-Llama-3-8B", **kwargs)


@register_model(tokenizer_args=TOKENIZER_ARGS)
def llama3_8b_embed(**kwargs):
    return create_embed_model("Meta-Llama-3-8B", **kwargs)


@register_model(**TOKENIZER_ARGS)
def llama3_8b_instruct_tokenizer(**kwargs):
    return create_tokenizer("Meta-Llama-3-8B-Instruct", **kwargs)


@register_model(tokenizer_args=TOKENIZER_ARGS)
def llama3_8b_instruct(**kwargs):
    return create_tokenizer_and_model("Meta-Llama-3-8B-Instruct", **kwargs)


@register_model(tokenizer_args=TOKENIZER_ARGS)
def llama3_8b_instruct_embed(**kwargs):
    return create_embed_model("Meta-Llama-3-8B-Instruct", **kwargs)


@register_model(**TOKENIZER_ARGS)
def llama3_70b_tokenizer(**kwargs):
    return create_tokenizer("Meta-Llama-3-70B", **kwargs)


@register_model(tokenizer_args=TOKENIZER_ARGS)
def llama3_70b(**kwargs):
    return create_tokenizer_and_model("Meta-Llama-3-70B", **kwargs)


@register_model(tokenizer_args=TOKENIZER_ARGS)
def llama3_70b_embed(**kwargs):
    return create_embed_model("Meta-Llama-3-70B", **kwargs)


@register_model(**TOKENIZER_ARGS)
def llama3_70b_instruct_tokenizer(**kwargs):
    return create_tokenizer("Meta-Llama-3-70B-Instruct", **kwargs)


@register_model(tokenizer_args=TOKENIZER_ARGS)
def llama3_70b_instruct(**kwargs):
    return create_tokenizer_and_model("Meta-Llama-3-70B-Instruct", **kwargs)


@register_model(tokenizer_args=TOKENIZER_ARGS)
def llama3_70b_instruct_embed(**kwargs):
    return create_embed_model("Meta-Llama-3-70B-Instruct", **kwargs)
