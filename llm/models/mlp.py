import torch.nn as nn

from .registry import register_model


def get_classifier(input_size=None, output_size=None, bias=False, **_):
    model = nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, output_size, bias=bias),
        # nn.Linear(input_size, output_size, bias=bias),
    )

    return model


@register_model
def mlp_binary(**kwargs):
    kwargs.pop("output_size", None)
    kwargs.pop("bias", None)
    return get_classifier(**kwargs, output_size=2, bias=True)
