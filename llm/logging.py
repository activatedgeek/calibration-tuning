from pathlib import Path
from uuid import uuid4
from functools import partial
import logging
import os
import json
import wandb
from accelerate import PartialState as AcceleratorState


WANDB_KWARGS_NAME = "wandb_args.json"


class WnBHandler(logging.Handler):
    """Listen for W&B logs.

    Default Usage:
    ```
    logging.log(metrics_dict, extra=dict(metrics=True, prefix='train'))
    ```

    `metrics_dict` (optionally prefixed) is directly consumed by `wandb.log`.
    """

    def emit(self, record):
        metrics = record.msg
        if hasattr(record, "prefix"):
            metrics = {f"{record.prefix}/{k}": v for k, v in metrics.items()}
        wandb.log(metrics)


class MetricsFilter(logging.Filter):
    def __init__(self, extra_key="metrics", invert=False):
        super().__init__()
        self.extra_key = extra_key
        self.invert = invert

    def filter(self, record):
        should_pass = hasattr(record, self.extra_key) and getattr(
            record, self.extra_key
        )
        if self.invert:
            should_pass = not should_pass
        return should_pass


def get_log_dir(log_dir=None):
    if log_dir is not None:
        return Path(log_dir)

    root_dir = (
        Path(os.environ.get("LOGDIR", Path.cwd() / ".log"))
        / Path.cwd().name
        / f"run-{str(uuid4())[:8]}"
    )
    log_dir = Path(str((root_dir / "files").resolve()))
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir


def set_logging(log_dir=None, metrics_extra_key="metrics", generate_log_dir=False):
    accelerator = AcceleratorState()

    log_dir = log_dir or os.environ.get(
        "WANDB_DIR", get_log_dir() if generate_log_dir else None
    )
    assert log_dir is not None, "Missing log_dir."

    with accelerator.main_process_first():
        os.makedirs(log_dir, exist_ok=True)

    if accelerator.is_main_process:
        ## Set other properties using environment variables: https://docs.wandb.ai/guides/track/environment-variables.
        wandb.init(
            mode=os.environ.get("WANDB_MODE", default="offline"),
            dir=log_dir,
            # settings=wandb.Settings(start_method="fork"),
        )
        ## Store sweep run config, if other processes need it.
        if "WANDB_SWEEP_ID" in os.environ:
            with open(f"{log_dir}/{WANDB_KWARGS_NAME}", "w") as f:
                json.dump(dict(wandb.config), f)

    _CONFIG = {
        "version": 1,
        "formatters": {
            "console": {
                "format": "[%(asctime)s] (%(funcName)s:%(levelname)s) %(message)s",
            },
        },
        "filters": {
            "metrics": {
                "()": MetricsFilter,
                "extra_key": metrics_extra_key,
            },
            "nometrics": {
                "()": MetricsFilter,
                "extra_key": metrics_extra_key,
                "invert": True,
            },
        },
        "handlers": {
            "null": {
                "()": logging.NullHandler,
            },
            "stdout": {
                "()": logging.StreamHandler,
                "formatter": "console",
                "stream": "ext://sys.stdout",
                "filters": ["nometrics"],
            },
            "wandb_file": {
                "()": WnBHandler,
                "filters": ["metrics"],
            },
        },
        "loggers": {
            "": {
                "handlers": ["stdout", "wandb_file"]
                if accelerator.is_main_process
                else ["null"],
                "level": os.environ.get("LOGLEVEL", "INFO"),
            },
        },
    }

    logging.config.dictConfig(_CONFIG)

    logging.info(f'Files stored in "{log_dir}".')

    def finish_logging():
        if accelerator.is_main_process:
            wandb.finish()

    return log_dir, finish_logging


def maybe_load_wandb_kwargs(path):
    wandb_kwargs_path = f"{path}/{WANDB_KWARGS_NAME}"
    if os.path.isfile(wandb_kwargs_path):
        with open(wandb_kwargs_path) as f:
            wandb_kwargs = json.load(f)
        return wandb_kwargs
    return {}


def entrypoint(main):
    accelerator = AcceleratorState()

    def _main(log_dir=None, **kwargs):
        with accelerator.main_process_first():
            log_dir, finish_logging = set_logging(log_dir=log_dir)
            kwargs = {**kwargs, **maybe_load_wandb_kwargs(log_dir)}

        if accelerator.is_main_process:
            logging.info(f"Working with {accelerator.num_processes} process(es).")

        main(**kwargs, log_dir=log_dir)

        finish_logging()

    def _entrypoint(**kwargs):
        if "WANDB_SWEEP_ID" in os.environ:
            if accelerator.is_main_process:
                wandb.agent(
                    os.environ.get("WANDB_SWEEP_ID"),
                    function=partial(_main, **kwargs),
                    count=1,
                )
            else:
                _main(**kwargs)
        else:
            _main(**kwargs)

    return _entrypoint
