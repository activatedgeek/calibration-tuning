import os
from functools import partial
import logging
from tqdm.auto import tqdm
import pandas as pd
from accelerate import Accelerator, PartialState as AcceleratorState
from peft import PeftModel

from llm.logging import set_logging, wandb, maybe_load_wandb_kwargs
from llm.datasets import list_datasets, get_dataset_attrs
from llm.models import get_model, get_special_tokens
from llm.utils.evaluation import evaluate_dataset
from llm.utils.trainer import get_last_checkpoint_path


def main(
    seed=137,
    log_dir=None,
    eval_kshot=None,
    data_dir=None,
    batch_size=1,
    model_name=None,
    fp8=True,
    model_dir=None,
    peft_dir=None,
    use_dataset_cache=True,
):
    accelerator = Accelerator()

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
        peft_dir = get_last_checkpoint_path(peft_dir)

        model = PeftModel.from_pretrained(model, peft_dir)

        logging.info(f"Loaded PEFT checkpoint from '{peft_dir}'")

    all_datasets = sorted(
        list(filter(lambda x: x != "mmlu", list_datasets()))
        + [f"mmlu:{task}" for task in get_dataset_attrs("mmlu").get("tasks")]
    )

    all_metrics = []
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
            use_cache=use_dataset_cache,
        )

        all_metrics += map(
            lambda m: {**m, **dict(wandb.config)},
            list(filter(lambda m: m is not None, [val_metrics, test_metrics])),
        )
        logging.info(
            {"metrics": wandb.Table(dataframe=pd.DataFrame(all_metrics))},
            extra=dict(metrics=True),
        )


def entrypoint(log_dir=None, **kwargs):
    accelerator = AcceleratorState()

    with accelerator.main_process_first():
        log_dir, finish_logging = set_logging(
            log_dir=log_dir, init_wandb=accelerator.is_main_process
        )
        kwargs = {**kwargs, **maybe_load_wandb_kwargs(log_dir)}

    if accelerator.is_main_process:
        logging.info(f"Working with {accelerator.num_processes} process(es).")

    main(**kwargs, log_dir=log_dir)

    finish_logging()


def pre_entrypoint(**kwargs):
    accelerator = AcceleratorState()

    if "WANDB_SWEEP_ID" in os.environ:
        if accelerator.is_main_process:
            wandb.agent(
                os.environ.get("WANDB_SWEEP_ID"),
                function=partial(entrypoint, **kwargs),
                count=1,
            )
        else:
            entrypoint(**kwargs)
    else:
        entrypoint(**kwargs)


if __name__ == "__main__":
    import fire

    fire.Fire(pre_entrypoint)
