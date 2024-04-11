import time
import logging
from functools import partial
from typing import Any
import numpy as np
import tiktoken
from openai import OpenAI, APIError
import torch

from .registry import register_model


class OpenAITokenizer:
    def __init__(self, model):
        self.encoding = tiktoken.encoding_for_model(model)

    def __call__(self, texts, return_tensors=None, return_length=True, **_) -> Any:
        input_ids = self.encoding.encode_batch(texts)

        out = dict(input_ids=input_ids)
        if return_length:
            out["length"] = [len(ids) for ids in input_ids]
            if return_tensors == "pt":
                out["length"] = torch.tensor(out["length"]).long()
        return out


def get_openai_tokenizer(model, **_):
    return OpenAITokenizer(model)


@register_model
def oai_gpt35t_tokenizer(*_, **kwargs):
    return get_openai_tokenizer("gpt-3.5-turbo", **kwargs)


@register_model
def oai_gpt4_tokenizer(*_, **kwargs):
    return get_openai_tokenizer("gpt-4", **kwargs)


class OpenAIEmbeddingModel:
    def __init__(self, model, dimension=1536, retries=10):
        client = OpenAI()
        self.retries = retries
        self.d = dimension
        self.model = partial(
            client.embeddings.create,
            model=model,
            encoding_format="float",
            dimensions=self.d,
        )

    def get_sentence_embedding_dimension(self):
        return self.d

    def encode(self, texts, **_):
        response = None

        __retries_left = int(self.retries)
        while response is None and __retries_left:
            try:
                response = self.model(input=texts)
            except APIError:
                logging.exception("OpenAI API Error.", exc_info=True)
                time.sleep(1)

            __retries_left -= 1

        embeddings = np.array([d.embedding for d in response.data])
        return embeddings

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


def get_openai_embedding_model(model, dimension=1536, retries=10, **_):
    return OpenAIEmbeddingModel(model, dimension=dimension, retries=retries)


@register_model
def oai_small(*_, **kwargs):
    return get_openai_embedding_model("text-embedding-3-small", **kwargs)
