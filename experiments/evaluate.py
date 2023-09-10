import logging
from tqdm.auto import tqdm
import wandb
import pandas as pd
import torch
from accelerate import Accelerator
from peft import PeftModel

from llm.logging import entrypoint
from llm.datasets import list_datasets, get_dataset_attrs
from llm.models import get_model
from llm.utils.evaluation import evaluate_dataset_via_eos
from llm.utils.trainer import get_last_checkpoint_path


def main(
    seed=137,
    log_dir=None,
    eval_kshot=None,
    dataset=None,
    data_dir=None,
    batch_size=1,
    model_name=None,
    model_dir=None,
    peft_dir=None,
    use_dataset_cache=True,
):
    accelerator = Accelerator()

    config = {
        "seed": seed,
        "model_name": model_name,
        "model_dir": model_dir,
        "peft_dir": peft_dir,
        "eval_kshot": eval_kshot,
    }
    if accelerator.is_main_process:
        wandb.config.update(config)

    tokenizer = get_model(
        f"{model_name}_tokenizer",
        model_dir=model_dir,
    )

    model = get_model(
        model_name,
        device_map="auto",
        # device_map={"": accelerator.local_process_index},
        torch_dtype=torch.float16,
        model_dir=model_dir,
        use_cache=False,
        tokenizer=tokenizer,
    )

    if peft_dir is not None:
        peft_dir = get_last_checkpoint_path(peft_dir)

        model = PeftModel.from_pretrained(model, peft_dir)

        logging.info(f"Loaded PEFT checkpoint from '{peft_dir}'")

    if dataset is None:
        all_datasets = sorted(
            list(
                filter(
                    lambda x: ("combined" not in x)
                    and ("mmlu" not in x)
                    and ("bbh" not in x),
                    list_datasets(),
                )
            )
            + [f"mmlu:{task}" for task in get_dataset_attrs("mmlu").get("tasks")]
        )
        logging.warning("No dataset argument used. Evaluating all datasets.")
    else:
        all_datasets = [dataset]

    all_metrics = []
    for dataset in tqdm(all_datasets):
        val_metrics, test_metrics = evaluate_dataset_via_eos(
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

        dataset_metrics = list(
            map(
                lambda m: {**m, **config, "dataset": dataset},
                list(filter(lambda m: m is not None, [val_metrics, test_metrics])),
            )
        )
        all_metrics += dataset_metrics
        logging.info(
            {"metrics": wandb.Table(dataframe=pd.DataFrame(all_metrics))},
            extra=dict(metrics=True),
        )


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint(main))
