from pathlib import Path
from datetime import datetime
from time import perf_counter
import logging.config
import os
import wandb

from .accelerate import Accelerator, AcceleratorState


class Timer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = perf_counter() - self.start


class WnBHandler(logging.Handler):
    """Listen for W&B logs.

    Default Usage:
    ```
    logging.log(metrics_dict, extra=dict(metrics=True, prefix='train'))
    ```

    `metrics_dict` (optionally prefixed) is directly consumed by `wandb.log`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.accelerator = AcceleratorState()

    def emit(self, record):
        metrics = record.msg
        if hasattr(record, "prefix"):
            metrics = {f"{record.prefix}/{k}": v for k, v in metrics.items()}
        if self.accelerator.is_main_process:
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


def setup_log_dir(log_dir=None):
    accelerator = AcceleratorState()

    if accelerator.is_main_process:
        if log_dir is None:
            root_dir = (
                Path(os.environ.get("LOGDIR", Path.home() / "logs")) / Path.cwd().name
            )
            dir_name = datetime.today().strftime("%Y-%m-%dT%H-%M-%S")
            if "SLURM_JOB_ID" in os.environ:
                dir_name = f"{os.environ.get('SLURM_JOB_ID')}-{dir_name}"

            log_dir = root_dir / dir_name
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(parents=True)
    else:
        log_dir = None

    log_dir = accelerator.sync_object(log_dir)

    return str(log_dir.resolve())


def setup_logging(log_dir=None, metrics_extra_key="metrics"):
    accelerator = AcceleratorState()

    log_dir = setup_log_dir(log_dir=log_dir)

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
                "handlers": (
                    ["stdout", "wandb_file"]
                    if accelerator.is_main_process
                    else ["null"]
                ),
                "level": os.environ.get("LOGLEVEL", "INFO"),
            },
        },
    }

    logging.config.dictConfig(_CONFIG)

    logging.info(f"Logging to '{log_dir}'.")

    return log_dir


def setup_wandb(log_dir):
    accelerator = AcceleratorState()

    wandb_kwargs = {}

    if accelerator.is_main_process:
        default_mode = (
            "online"
            if any([k in os.environ for k in ["WANDB_RUN_ID", "WANDB_SWEEP_ID"]])
            else "offline"
        )

        wandb.init(
            dir=log_dir,
            mode=os.environ.get("WANDB_MODE", default_mode),
            project=os.environ.get("WANDB_PROJECT", Path().cwd().name),
            entity=os.environ.get("WANDB_ENTITY"),
            name=os.environ.get("WANDB_NAME", Path(log_dir).name),
            resume="WANDB_RUN_ID" in os.environ,
            id=os.environ.get("WANDB_RUN_ID"),
        )

        run = wandb.run
        ## Get arguments for sweep run ID.
        if "WANDB_RUN_ID" in os.environ:
            run = wandb.Api().run(
                "/".join(
                    [
                        wandb.run.entity,
                        wandb.run.project,
                        os.environ.get("WANDB_RUN_ID"),
                    ]
                )
            )

        wandb_kwargs = {
            **wandb_kwargs,
            **{k: v for k, v in run.config.items() if v is not None},
        }

    wandb_kwargs = accelerator.sync_object(wandb_kwargs)

    return wandb_kwargs


def entrypoint(main):
    def _wrapped_main(log_dir=None, **kwargs):
        log_dir = setup_logging(log_dir=log_dir)

        wandb_kwargs = setup_wandb(log_dir)

        kwargs = {**wandb_kwargs, **kwargs, "log_dir": log_dir}

        return main(**kwargs)

    def _wrapped_entrypoint(**kwargs):
        accelerator = Accelerator()

        if "WANDB_SWEEP_ID" in os.environ:
            if accelerator.is_main_process:
                wandb.agent(
                    os.environ.get("WANDB_SWEEP_ID"),
                    project=os.environ.get("WANDB_PROJECT", Path().cwd().name),
                    entity=os.environ.get("WANDB_ENTITY"),
                    function=lambda **agent_kwargs: _wrapped_main(
                        **agent_kwargs,
                        **kwargs,
                    ),
                    count=1,
                )
            else:
                _wrapped_main(**kwargs)
        else:
            _wrapped_main(**kwargs)

    return _wrapped_entrypoint
