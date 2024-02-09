import os
import logging
import torch


from .. import get_model
from .utils import get_last_checkpoint


def get_classifier_head(
    model, checkpoint_dir=None, is_trainable=False, weights_name="classifier_model.bin"
):
    classifier_model = get_model(
        "mlp_binary", input_size=model.config.hidden_size, output_size=2
    )

    if checkpoint_dir is not None:
        checkpoint_dir = get_last_checkpoint(checkpoint_dir)

        if os.path.isfile(f"{checkpoint_dir}/{weights_name}"):
            classifier_model.load_state_dict(
                torch.load(f"{checkpoint_dir}/{weights_name}")
            )

            logging.info(f"Loaded classifier model checkpoint from '{checkpoint_dir}'.")

    if is_trainable:
        classifier_model = classifier_model.train().requires_grad_(True)
    else:
        classifier_model = classifier_model.eval().requires_grad_(False)

    return classifier_model
