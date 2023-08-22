import os
import logging
from accelerate import Accelerator
from peft import PeftModel

from llm.logging import set_logging, wandb
from llm.datasets import get_dataset, get_loader
from llm.datasets.llm_utils import DataCollatorForSupervisedDataset
from llm.models import get_model, get_special_tokens
from llm.utils.distributed import WaitForMainProcess
from llm.utils.evaluation import evaluate_via_eos


def main(
    accelerator,
    seed=137,
    log_dir=None,
    dataset=None,
    dataset_instance=None,
    data_dir=None,
    batch_size=1,
    model_name=None,
    fp8=True,
    model_dir=None,
    peft_dir=None,
):
    if accelerator.is_main_process:
        wandb.config.update(
            {
                "seed": seed,
                "dataset": dataset,
                "dataset_instance": dataset_instance,
                "model_name": model_name,
                "fp8": fp8,
                "model_dir": model_dir,
                "peft_dir": peft_dir,
            }
        )

    tokenizer = get_model(
        f"{model_name}_tokenizer",
        model_dir=model_dir,
    )
    special_token_count = tokenizer.add_special_tokens(get_special_tokens(tokenizer))

    with WaitForMainProcess(accelerator):
        _, val_data, test_data = get_dataset(
            dataset,
            instance=dataset_instance,
            root=data_dir,
            tokenizer=tokenizer,
            seed=seed,
        )

    model = get_model(
        model_name,
        device_map="auto",
        load_in_8bit=fp8,
        model_dir=model_dir,
    )

    model.resize_token_embeddings(len(tokenizer))
    if special_token_count:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings[-special_token_count:] = input_embeddings[
            :-special_token_count
        ].mean(dim=0, keepdim=True)
        output_embeddings[-special_token_count:] = output_embeddings[
            :-special_token_count
        ].mean(dim=0, keepdim=True)

    if peft_dir is not None:
        model = PeftModel.from_pretrained(model, peft_dir)

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

    val_metrics = _evaluate(val_data)
    logging.info(val_metrics, extra=dict(metrics=True, prefix="val"))

    test_metrics = _evaluate(test_data)
    logging.info(test_metrics, extra=dict(metrics=True, prefix="test"))


def entrypoint(log_dir=None, **kwargs):
    accelerator = Accelerator()

    ## Only setup logging from one process.
    log_dir, finish_logging = (
        set_logging(log_dir=os.environ.get("WANDB_DIR", log_dir))
        if accelerator.is_main_process
        else [None, None]
    )
    if accelerator.is_main_process:
        logging.info(f"Working with {accelerator.num_processes} process(es).")

    main(accelerator, **kwargs, log_dir=log_dir)

    if accelerator.is_main_process:
        finish_logging()


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint)
