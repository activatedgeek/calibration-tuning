import os
import logging
from accelerate import Accelerator

from llm.logging import set_logging, wandb
from llm.datasets import get_dataset, get_loader
from llm.datasets.llm_utils import DataCollatorForSupervisedDataset
from llm.models import create_model, get_special_tokens
from llm.utils.evaluation import evaluate_via_eos


## FIXME: load from checkpoint.
def main(
    accelerator,
    seed=None,
    log_dir=None,
    data_dir=None,
    model_dir=None,
    dataset=None,
    dataset_instance=None,
    batch_size=1,
    model_name=None,
    fp8=True,
):
    if accelerator.is_main_process:
        wandb.config.update({
            "dataset": dataset,
            "dataset_instance": dataset_instance,
            "model_name": model_name,
            "fp8": fp8,
        })

    tokenizer = create_model(
        model_name=f"{model_name}_tokenizer", model_kwargs=dict(cache_dir=model_dir)
    )
    tokenizer.add_special_tokens(get_special_tokens(tokenizer))

    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()

    _, val_data, test_data = get_dataset(
        dataset,
        instance=dataset_instance,
        root=data_dir,
        tokenizer=tokenizer,
        seed=seed,
    )

    if accelerator.is_main_process:
        accelerator.wait_for_everyone()

    model = create_model(
        model_name=model_name,
        model_kwargs=dict(
            device_map={"": accelerator.device},
            load_in_8bit=fp8,
            cache_dir=model_dir,
        ),
    )

    def _evaluate(_data):
        return evaluate_via_eos(
            accelerator,
            model,
            tokenizer,
            get_loader(
                _data,
                batch_size=batch_size,
                collate_fn=DataCollatorForSupervisedDataset(tokenizer),
                accelerator=accelerator,
            ),
        )

    # train_metrics = _evaluate(train_data)
    # logging.info(train_metrics, extra=dict(metrics=True, prefix="train"))

    val_metrics = _evaluate(val_data)
    logging.info(val_metrics, extra=dict(metrics=True, prefix="val"))

    test_metrics = _evaluate(test_data)
    logging.info(test_metrics, extra=dict(metrics=True, prefix="test"))


def entrypoint(seed=None, log_dir=None, **kwargs):
    accelerator = Accelerator()

    ## Only setup logging from one process.
    log_dir, finish_logging = (
        set_logging(log_dir=os.environ.get("WANDB_DIR", log_dir)) if accelerator.is_main_process else [None, None]
    )
    if accelerator.is_main_process:
        logging.info(f"Working with {accelerator.num_processes} process(es).")

    main(accelerator, **kwargs, seed=seed, log_dir=log_dir)

    if accelerator.is_main_process:
        finish_logging()


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint)
