import os
import logging
from tqdm.auto import tqdm
import wandb
import pandas as pd
import torch

from llm.datasets import get_dataset_attrs, get_dataset
from llm.eval import evaluate_dataset
from llm.logging import entrypoint
from llm.models import get_model
from llm.models.peft import (
    get_lora_model,
    get_classifier_head,
    get_temperature_head,
    get_temperature_scale_model,
)
from llm.models.peft.utils import get_last_checkpoint_path
from llm.trainer import ClassificationTuner, CalibrationTuner, FineTuner


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
    embedding_model_name=None,
    model_name=None,
    peft_dir=None,
    query_peft_dir=None,
    scale_temp=None,
    with_classifier=False,
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
        peft_dir=peft_dir,
        query_peft_dir=query_peft_dir,
        scale_temp=scale_temp,
        with_classifier=with_classifier,
        mode=mode,
        batch_size=batch_size,
    )
    if accelerator.is_main_process:
        wandb.config.update(config, allow_val_change=True)

    tokenizer, model = get_model(model_name, device_map="auto")

    model = get_lora_model(
        model,
        peft_id_or_dir=peft_dir,
        is_trainable=False,
        adapter_name="default",
    )

    model = get_lora_model(
        model,
        peft_id_or_dir=query_peft_dir,
        is_trainable=False,
        adapter_name="query",
    )

    if with_classifier:
        if embedding_model_name is not None:
            model.embedding_model = get_model(embedding_model_name)

        classifier_model = get_classifier_head(
            input_size=(
                model.embedding_model.get_sentence_embedding_dimension()
                if embedding_model_name
                else model.config.hidden_size
            ),
            checkpoint_dir=None if scale_temp == "probe" else query_peft_dir,
            is_trainable=False,
            weights_name=ClassificationTuner.WEIGHTS_NAME,
        )

        if scale_temp == "probe":
            temperature_model = get_temperature_head()

            classifier_model = torch.nn.Sequential(
                classifier_model,
                temperature_model,
            )

            if query_peft_dir is not None:
                checkpoint_dir = get_last_checkpoint_path(query_peft_dir)

                if os.path.isfile(
                    f"{checkpoint_dir}/{ClassificationTuner.WEIGHTS_NAME}"
                ):
                    classifier_model.load_state_dict(
                        torch.load(
                            f"{checkpoint_dir}/{ClassificationTuner.WEIGHTS_NAME}"
                        )
                    )

                    logging.info(
                        f"Loaded temperature-scaled classifier model checkpoint from '{checkpoint_dir}'."
                    )

        model.classifier_model = classifier_model.to(model.dtype)
        model.classifier_model = model.classifier_model.to(accelerator.device)
        model.classifier_model.target_layer = -1

    if scale_temp == "logits":
        ## @NOTE: Only for fine-tuned models.
        model = get_temperature_scale_model(
            model,
            checkpoint_dir=peft_dir,
            is_trainable=True,
            weights_name=FineTuner.TEMPERATURE_WEIGHTS_NAME,
        )
    elif scale_temp == "query":
        ## @NOTE: Only for calibration-tuned models.
        temperature_model = get_temperature_head(
            checkpoint_dir=query_peft_dir or peft_dir,
            is_trainable=False,
            weights_name=CalibrationTuner.TEMPERATURE_WEIGHTS_NAME,
        ).to(accelerator.local_process_index)

        model.query_temperature_model = temperature_model
    elif scale_temp == "probe":
        ## @NOTE: Already handled earlier.
        pass
    else:
        if scale_temp is not None:
            raise NotImplementedError

    # model = torch.nn.Sequential(
    #     get_classifier_head(
    #         input_size=4096,
    #         is_trainable=False,
    #     ),
    #     get_temperature_head(is_trainable=False),
    # ).to(accelerator.device)

    # model = get_classifier_head(
    #     input_size=8192,
    #     is_trainable=False,
    # ).to(accelerator.device)

    # if query_peft_dir is not None:
    #     ckpt_path = f"{get_last_checkpoint_path(query_peft_dir)}/{ClassificationTuner.WEIGHTS_NAME}"
    #     model.load_state_dict(torch.load(ckpt_path))

    #     logging.info(f"Loaded checkpoint from {ckpt_path}")

    model.eval()

    if get_dataset_attrs(dataset).get("collection", False):
        all_datasets = get_dataset(dataset)
    else:
        assert dataset is not None, "Missing dataset."
        all_datasets = [dataset]

    all_metrics = []
    for dataset in tqdm(all_datasets):
        metrics = evaluate_dataset(
            accelerator,
            model,
            tokenizer,
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

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.save(f"{log_dir}/metrics/*", base_path=log_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
