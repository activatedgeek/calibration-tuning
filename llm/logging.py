from pathlib import Path
from datetime import datetime
from time import perf_counter
import logging
import logging.config
import os
import wandb

from .distributed import Accelerator, AcceleratorState


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
            metrics = {f"{record.prefix}{k}": v for k, v in metrics.items()}
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


def setup_logging(log_dir=None, metrics_extra_key="metrics", is_main_process=True):
    if is_main_process:
        if log_dir is None:
            log_dir_name = datetime.today().strftime("%Y-%m-%dT%H-%M-%S")
            if "JOB_ID" in os.environ:
                log_dir_name = os.getenv("JOB_ID") + "-" + log_dir_name
            log_dir = (
                Path(os.getenv("PROJECT_HOME", Path.home()))
                / Path.cwd().name
                / "logs"
                / log_dir_name
            )
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(parents=True)
        log_dir = str(log_dir.resolve())

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
                "handlers": (["stdout", "wandb_file"] if is_main_process else ["null"]),
                "level": os.getenv("LOGLEVEL", "INFO"),
            },
        },
    }

    logging.config.dictConfig(_CONFIG)

    if is_main_process:
        logging.info(f"Logging to '{log_dir}'.")

    return log_dir


def setup_wandb(log_dir, return_run_id=False):
    os.environ[wandb.env.SILENT] = "true"

    wandb_kwargs = {}

    wandb.init(
        dir=log_dir,
        mode=os.getenv(
            wandb.env.MODE,
            (
                "online"
                if any(
                    [k in os.environ for k in [wandb.env.RUN_ID, wandb.env.SWEEP_ID]]
                )
                else "offline"
            ),
        ),
        # entity=os.getenv(wandb.env.ENTITY),
        project=os.getenv(wandb.env.PROJECT, Path().cwd().name),
        name=os.getenv(wandb.env.NAME, Path(log_dir).name),
        # id=os.getenv(wandb.env.RUN_ID),
        resume=wandb.env.RUN_ID in os.environ,
        allow_val_change=any(
            [k in os.environ for k in [wandb.env.RUN_ID, wandb.env.SWEEP_ID]]
        ),
    )

    run = wandb.run
    ## Get arguments for sweep run ID.
    if wandb.env.RUN_ID in os.environ:
        run = wandb.Api().run(
            "/".join(
                [
                    wandb.run.entity,
                    wandb.run.project,
                    os.getenv(wandb.env.RUN_ID),
                ]
            )
        )

    wandb_kwargs = {
        **wandb_kwargs,
        **{k: v for k, v in run.config.items() if v is not None},
    }

    if return_run_id:
        if run.url:
            logging.info(f"View run at {run.url}")
        return run.id, wandb_kwargs

    return wandb_kwargs


def entrypoint(main=None, with_accelerator=False, with_logging=True, with_wandb=True):
    def _decorator(f):
        def _wrapped_entrypoint(**kwargs):
            accelerator = Accelerator()

            if with_accelerator:
                kwargs["accelerator"] = accelerator

            if with_logging:
                kwargs["log_dir"] = setup_logging(
                    log_dir=kwargs.get("log_dir", None),
                    is_main_process=accelerator.is_main_process,
                )

            if with_wandb and accelerator.is_main_process:
                ##
                # NOTE: To avoid multiprocessing conflicts,
                # get config from sweep agent, stop, and resume with the run ID.
                #
                if wandb.env.SWEEP_ID in os.environ:
                    logging.info("Getting config from sweep agent.")

                    run_id = None

                    def _agent_dummy_init(**agent_kwargs):
                        nonlocal kwargs, run_id

                        run_id, wandb_kwargs = setup_wandb(
                            kwargs["log_dir"], return_run_id=True
                        )
                        kwargs = {**wandb_kwargs, **agent_kwargs, **kwargs}

                    wandb.agent(
                        os.getenv(wandb.env.SWEEP_ID),
                        project=os.getenv(wandb.env.PROJECT, Path().cwd().name),
                        # entity=os.getenv(wandb.env.ENTITY),
                        function=_agent_dummy_init,
                        count=1,
                    )

                    os.environ[wandb.env.RUN_ID] = run_id

                kwargs = {**setup_wandb(kwargs["log_dir"]), **kwargs}

            kwargs = accelerator.sync_object(kwargs)
            return f(**kwargs)

        return _wrapped_entrypoint

    if main:
        return _decorator(main)
    return _decorator
