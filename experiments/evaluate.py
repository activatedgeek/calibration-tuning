import logging
from tqdm.auto import tqdm
import wandb
import pandas as pd
import torch
from accelerate import Accelerator

from llm.logging import entrypoint, Timer
from llm.datasets import get_all_datasets_list
from llm.models import get_model
from llm.models.peft import get_lora_model, prepare_model_for_temperature_scaling
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
    scale_temp=False,
    use_dataset_cache=True,
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
        "log_dir": log_dir,
    }
    if accelerator.is_main_process:
        wandb.config.update(config)

    tokenizer = get_model(
        f"{model_name}_tokenizer",
        model_dir=model_dir,
    )

    model = get_model(
        model_name,
        device_map={"": accelerator.local_process_index},
        torch_dtype=torch.float16,
        model_dir=model_dir,
        use_cache=False,
        tokenizer=tokenizer,
    )

    model = get_lora_model(
        model,
        peft_dir=peft_dir,
        is_trainable=False,
        adapter_name="default",
    )

    model = get_lora_model(
        model,
        peft_dir=query_peft_dir or peft_dir,
        is_trainable=False,
        adapter_name="query",
    ).to(accelerator.local_process_index)

    if scale_temp:
        prepare_model_for_temperature_scaling(
            model, peft_dir=query_peft_dir or peft_dir
        )

    model.eval()

    if dataset.startswith("eval"):
        all_datasets = get_all_datasets_list(dataset)
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
                log_dir=log_dir,
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

        accelerator.free_memory()


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint(main))
