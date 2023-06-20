import logging
import yaml
from pathlib import Path
import timm
import torch
import torch.nn as nn


__BEST_CKPT_NAME = "best_model.pt"


def create_model(
    cfg_path=None,
    num_classes=None,
    in_chans=None,
    model_name=None,
    transfer=False,
    model_kwargs=None,
):
    ## Prepare configurations.
    model_cfg, base_ckpt_path = None, None

    if cfg_path is not None:
        with open(cfg_path, "r") as f:
            model_cfg = yaml.safe_load(f)

        net_ckpt_file = model_cfg.get("ckpt_file", __BEST_CKPT_NAME)
        base_ckpt_path = Path(cfg_path).parent / net_ckpt_file
    else:
        model_cfg = dict(
            model_name=model_name, num_classes=num_classes, in_chans=in_chans
        )

    ## Load model.
    model = timm.create_model(
        **model_cfg, **(model_kwargs or dict()), checkpoint_path=base_ckpt_path
    )
    if base_ckpt_path is not None:
        logging.info(f'Loaded base model from "{base_ckpt_path}".')

    ## Replace classifier head for transfer learning.
    if transfer:
        model.reset_classifier(num_classes)
        model_cfg["num_classes"] = num_classes
        logging.info(f"Reset classifier for {num_classes} classes.")

    return model


def save_model(model, save_path):
    torch.save(
        (
            model.module
            if isinstance(model, nn.parallel.DistributedDataParallel)
            else model
        ).state_dict(),
        save_path,
    )
