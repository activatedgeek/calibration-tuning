import logging
from tqdm.auto import tqdm
import wandb
import pandas as pd
import torch
from accelerate import Accelerator

from llm.logging import entrypoint, Timer
from llm.datasets import get_all_train_datasets, get_all_eval_datasets
from llm.models import get_model, load_peft_model_from_pretrained
from llm.models.peft import get_temperature_scaled_model
from llm.eval import evaluate_dataset


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
    query_peft_dir=None,
    use_dataset_cache=True,
    use_auto_device=False,
    prompt_style="choice",
    mode=None,
):
    accelerator = Accelerator()

    config = {
        "seed": seed,
        "model_name": model_name,
        "model_dir": model_dir,
        "peft_dir": peft_dir,
        "query_peft_dir": query_peft_dir,
        "eval_kshot": eval_kshot,
        "prompt_style": prompt_style,
        "mode": mode,
    }
    if accelerator.is_main_process:
        wandb.config.update(config)

    tokenizer = get_model(
        f"{model_name}_tokenizer",
        model_dir=model_dir,
    )

    model = get_model(
        model_name,
        device_map="auto" if use_auto_device else {"": accelerator.local_process_index},
        torch_dtype=torch.float16,
        model_dir=model_dir,
        use_cache=False,
        tokenizer=tokenizer,
    )

    model = load_peft_model_from_pretrained(
        model, peft_dir=peft_dir, query_peft_dir=query_peft_dir
    )

    model = get_temperature_scaled_model(
        model, checkpoint_dir=query_peft_dir or peft_dir
    )

    if dataset == "all":
        all_datasets = get_all_train_datasets() + get_all_eval_datasets()
    elif dataset == "eval":
        all_datasets = get_all_eval_datasets()
    else:
        assert dataset is not None, "Missing dataset."
        all_datasets = [dataset]

    all_metrics = []
    for dataset in tqdm(all_datasets):
        with Timer() as t:
            val_metrics, test_metrics = evaluate_dataset(
                accelerator,
                model,
                tokenizer,
                dataset,
                train_data=False,
                seed=seed,
                batch_size=batch_size,
                data_dir=data_dir,
                eval_kshot=eval_kshot,
                use_cache=use_dataset_cache,
                prompt_style=prompt_style,
                evaluate_fn=mode,
            )

        dataset_metrics = list(
            map(
                lambda m: {**m, **config, "dataset": dataset, "ts": t.elapsed},
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
