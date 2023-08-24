import os
import logging
from tqdm.auto import tqdm
from accelerate import Accelerator
from peft import PeftModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint

from llm.logging import set_logging, wandb
from llm.datasets import get_dataset, get_loader, list_datasets, get_dataset_attrs
from llm.datasets.llm_utils import DataCollatorForSupervisedDataset
from llm.models import get_model, get_special_tokens
from llm.utils.evaluation import evaluate_via_eos


def evaluate_dataset(
    accelerator,
    model,
    tokenizer,
    dataset,
    seed=137,
    batch_size=1,
    data_dir=None,
    eval_kshot=None,
):
    with accelerator.main_process_first():
        _extra_args = dict()
        ## NOTE: Conditional to avoid overriding default kshot specification in dataset definition.
        if eval_kshot is not None:
            _extra_args["eval_kshot"] = eval_kshot
        _, val_data, test_data = get_dataset(
            dataset, root=data_dir, tokenizer=tokenizer, seed=seed, **_extra_args
        )

    val_metrics = None
    if val_data is not None:
        val_metrics = evaluate_via_eos(
            accelerator,
            model,
            tokenizer,
            get_loader(
                val_data,
                batch_size=batch_size,
                collate_fn=DataCollatorForSupervisedDataset(tokenizer),
                accelerator=accelerator,
            ),
        )

    test_metrics = None
    if test_data is not None:
        test_metrics = evaluate_via_eos(
            accelerator,
            model,
            tokenizer,
            get_loader(
                test_data,
                batch_size=batch_size,
                collate_fn=DataCollatorForSupervisedDataset(tokenizer),
                accelerator=accelerator,
            ),
        )

    return val_metrics, test_metrics


def main(
    accelerator,
    seed=137,
    log_dir=None,
    eval_kshot=None,
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
                "model_name": model_name,
                "fp8": fp8,
                "model_dir": model_dir,
                "peft_dir": peft_dir,
                "eval_kshot": eval_kshot,
            }
        )

    tokenizer = get_model(
        f"{model_name}_tokenizer",
        model_dir=model_dir,
    )
    special_token_count = tokenizer.add_special_tokens(get_special_tokens(tokenizer))

    model = get_model(
        model_name,
        device_map={"": accelerator.local_process_index},
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
        if PREFIX_CHECKPOINT_DIR not in peft_dir:
            peft_dir = get_last_checkpoint(peft_dir)

            assert peft_dir is not None, f"No checkpoint found in '{peft_dir}'."

        model = PeftModel.from_pretrained(model, peft_dir)

        logging.info(f"Loaded PEFT checkpoint from '{peft_dir}'")

    all_datasets = list(filter(lambda x: x != "mmlu", list_datasets())) + [
        f"mmlu:{task}" for task in get_dataset_attrs("mmlu").get("tasks")
    ]

    for dataset in tqdm(all_datasets):
        val_metrics, test_metrics = evaluate_dataset(
            accelerator,
            model,
            tokenizer,
            dataset,
            seed=seed,
            batch_size=batch_size,
            data_dir=data_dir,
            eval_kshot=eval_kshot,
        )
        if val_metrics is not None:
            logging.info(val_metrics, extra=dict(metrics=True, prefix=f"{dataset}/val"))
        if test_metrics is not None:
            logging.info(
                test_metrics, extra=dict(metrics=True, prefix=f"{dataset}/test")
            )


def entrypoint(log_dir=None, **kwargs):
    log_dir = os.environ.get("WANDB_DIR", log_dir)

    accelerator = Accelerator()

    ## Only setup logging from one process.
    log_dir, finish_logging = (
        set_logging(log_dir) if accelerator.is_main_process else [None, None]
    )
    if accelerator.is_main_process:
        logging.info(f"Working with {accelerator.num_processes} process(es).")

    main(accelerator, **kwargs, log_dir=log_dir)

    if accelerator.is_main_process:
        finish_logging()


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint)
