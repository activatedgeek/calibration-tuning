import logging
from tqdm.auto import tqdm
import wandb
import pandas as pd

from llm.datasets import get_dataset_attrs, get_dataset
from llm.eval import evaluate_dataset
from llm.logging import entrypoint


@entrypoint(with_accelerator=True)
def main(
    accelerator=None,
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    prompt_style=None,
    eval_kshot=None,
    use_dataset_cache=True,
    model_name=None,
    scale_temp=True,
    mode=None,
    batch_size=1,
):
    config = dict(
        seed=seed,
        log_dir=log_dir,
        dataset=dataset,
        prompt_style=prompt_style,
        eval_kshot=eval_kshot,
        use_dataset_cache=use_dataset_cache,
        model_name=model_name,
        scale_temp=scale_temp,
        mode=mode,
        batch_size=batch_size,
    )
    if accelerator.is_main_process:
        wandb.config.update(config)

    if get_dataset_attrs(dataset).get("collection", False):
        all_datasets = get_dataset(dataset)
    else:
        assert dataset is not None, "Missing dataset."
        all_datasets = [dataset]

    all_metrics = []
    for dataset in tqdm(all_datasets):
        metrics = evaluate_dataset(
            accelerator,
            None,
            None,
            dataset,
            data_dir=data_dir,
            prompt_style=prompt_style,
            eval_kshot=eval_kshot,
            use_cache=use_dataset_cache,
            train_data=False,
            seed=seed,
            batch_size=batch_size,
            log_dir=log_dir,
            evaluate_fn=mode,
        )

        all_metrics += metrics
        logging.info(
            {"metrics": wandb.Table(dataframe=pd.DataFrame(all_metrics))},
            extra=dict(metrics=True),
        )

        accelerator.free_memory()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
