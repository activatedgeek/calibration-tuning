from sentence_transformers import SentenceTransformer

from .registry import register_model


def get_mpnet(**_):
    return SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")


@register_model
def mpnet_mqa(*args, **kwargs):
    return get_mpnet(*args, **kwargs)
