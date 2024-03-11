import os
import logging
from tqdm.auto import tqdm
import wandb
import pandas as pd
import torch

from llm.accelerate import Accelerator
from llm.logging import entrypoint
from llm.datasets import get_all_datasets_list
from llm.models import get_model
from llm.models.peft import (
    get_lora_model,
    get_classifier_head,
    get_temperature_head,
    get_temperature_scale_model,
)
from llm.models.peft.utils import get_last_checkpoint_path
from llm.eval import evaluate_dataset
from llm.trainer import ClassificationTuner, CalibrationTuner, FineTuner


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
    with_classifier=False,
    scale_temp=None,
    use_dataset_cache=True,
    prompt_style="choice",
    mode=None,
    int8=False,
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
        "int8": int8,
        "dataset": dataset,
        "data_dir": data_dir,
        "batch_size": batch_size,
        "scale_temp": scale_temp,
        "with_classifier": with_classifier,
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
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        model_dir=model_dir,
        use_cache=False,
        tokenizer=tokenizer,
        load_in_8bit=int8,
    )

    model = get_lora_model(
        model,
        peft_dir=peft_dir,
        is_trainable=False,
        adapter_name="default",
    )

    if query_peft_dir:
        model = get_lora_model(
            model,
            peft_dir=query_peft_dir,
            is_trainable=False,
            adapter_name="query",
        )

        if with_classifier:
            classifier_model = get_classifier_head(
                model,
                checkpoint_dir=None if scale_temp == "lora-probe" else query_peft_dir,
                is_trainable=False,
                weights_name=ClassificationTuner.WEIGHTS_NAME,
            )

            if scale_temp == "lora-probe":
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

            model.classifier_model = classifier_model.to(accelerator.device)
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
    elif scale_temp == "lora-probe":
        ## @NOTE: Already handled earlier.
        pass
    else:
        if scale_temp is not None:
            raise NotImplementedError

    model.eval()

    if dataset.startswith("eval"):
        all_datasets = get_all_datasets_list(dataset)
    else:
        assert dataset is not None, "Missing dataset."
        all_datasets = [dataset]

    all_metrics = []
    for dataset in tqdm(all_datasets):
        try:
            metrics = evaluate_dataset(
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

            all_metrics += metrics
            logging.info(
                {"metrics": wandb.Table(dataframe=pd.DataFrame(all_metrics))},
                extra=dict(metrics=True),
            )
        except torch.cuda.OutOfMemoryError:
            logging.exception(f"OOM fail for {dataset}.", exc_info=True)

        accelerator.free_memory()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.save(f"{log_dir}/metrics/*", base_path=log_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint(main))
