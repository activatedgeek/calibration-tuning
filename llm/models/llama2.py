from peft import prepare_model_for_kbit_training
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from ..datasets import LabeledStringDataCollator
from .registry import register_model
from .llm_model_utils import DEFAULT_PAD_TOKEN, resize_token_embeddings


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
def llama2_7b_tokenizer(**kwargs):
    return create_tokenizer("Llama-2-7b-hf", **kwargs)


@register_model
def llama2_7b(**kwargs):
    return create_tokenizer_and_model("Llama-2-7b-hf", **kwargs)


@register_model
def llama2_7b_embed(**kwargs):
    return create_embed_model("Llama-2-7b-hf", **kwargs)


@register_model
def llama2_7b_chat_tokenizer(**kwargs):
    return create_tokenizer("Llama-2-7b-chat-hf", **kwargs)


@register_model
def llama2_7b_chat(**kwargs):
    return create_tokenizer_and_model("Llama-2-7b-chat-hf", **kwargs)


@register_model
def llama2_13b_tokenizer(**kwargs):
    return create_tokenizer("Llama-2-13b-hf", **kwargs)


@register_model
def llama2_13b(**kwargs):
    return create_tokenizer_and_model("Llama-2-13b-hf", **kwargs)


@register_model
def llama2_13b_chat_tokenizer(**kwargs):
    return create_tokenizer("Llama-2-13b-chat-hf", **kwargs)


@register_model
def llama2_13b_chat(**kwargs):
    return create_tokenizer_and_model("Llama-2-13b-chat-hf", **kwargs)


@register_model
def llama2_70b_tokenizer(**kwargs):
    return create_tokenizer("Llama-2-70b-hf", **kwargs)


@register_model
def llama2_70b(**kwargs):
    return create_tokenizer_and_model("Llama-2-70b-hf", **kwargs)


@register_model
def llama2_70b_chat_tokenizer(**kwargs):
    return create_tokenizer("Llama-2-70b-chat-hf", **kwargs)


@register_model
def llama2_70b_chat(**kwargs):
    return create_tokenizer_and_model("Llama-2-70b-chat-hf", **kwargs)
