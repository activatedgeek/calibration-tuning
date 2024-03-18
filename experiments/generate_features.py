import os
import wandb
import torch
from tqdm.auto import tqdm
from peft import PeftModel

import multiprocess.context as ctx

## @HOTFIX: for hanging processes in dataset map.
# ctx._force_start_method("spawn")

from llm.datasets import (
    get_dataset,
    get_loader,
    prepare_uncertainty_query,
    LabeledStringDataCollator,
)
from llm.distributed import Accelerator
from llm.logging import entrypoint
from llm.models import get_model
from llm.models.peft import get_lora_model


@entrypoint
def main(
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    batch_size=1,
    model_name=None,
    peft_dir=None,
    query_peft_dir=None,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    use_dataset_cache=True,
    prompt_style=None,
    kshot=0,
    max_new_tokens=30,
    int8=False,
):
    accelerator = Accelerator()

    config = {
        "seed": seed,
        "model_name": model_name,
        "peft_dir": peft_dir,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "prompt_style": prompt_style,
        "max_new_tokens": max_new_tokens,
        "log_dir": log_dir,
        "int8": int8,
    }
    if accelerator.is_main_process:
        wandb.config.update(config)

    tokenizer, model = get_model(
        model_name,
        device_map={"": accelerator.local_process_index},
        use_int8=int8,
    )

    model = get_lora_model(
        model,
        peft_dir=peft_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        is_trainable=False,
        adapter_name="default",
    )

    if query_peft_dir is not None:
        model = get_lora_model(
            model,
            peft_dir=query_peft_dir,
            is_trainable=False,
            adapter_name="query",
        )

    model.eval()

    with accelerator.main_process_first():
        data_splits = get_dataset(
            dataset,
            root=data_dir,
            tokenizer=tokenizer,
            seed=seed,
            num_workers=16,  # os.cpu_count() // 2,
            use_cache=use_dataset_cache,
            prompt_style=prompt_style,
            train_kshot=kshot,
            eval_kshot=kshot,
        )
        data_splits = [
            (s, ds)
            for s, ds in zip(["train", "validation", "test"], data_splits)
            if ds is not None
        ]

    for split_name, data in data_splits:
        loader = get_loader(
            data,
            batch_size=batch_size,
            pin_memory=True,
            accelerator=accelerator,
        )

        with accelerator.main_process_first():
            if accelerator.is_main_process:
                os.makedirs(f"{log_dir}/outputs/{split_name}")

        collate_fn = LabeledStringDataCollator(tokenizer)

        all_features, all_labels = [], []

        for inputs in tqdm(loader):
            inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
            targets = [inp.pop("target") for inp in inputs]
            outputs = [inp.pop("output") for inp in inputs]
            query_labels = [inp.pop("query_label") for inp in inputs]

            if isinstance(model, PeftModel) and "query" in model.peft_config:
                model.set_adapter("query")

                feature_inputs, _, _ = prepare_uncertainty_query(
                    tokenizer,
                    inputs,
                    targets,
                    outputs,
                    strategy="substring",
                    query_labels=query_labels,
                )
            else:
                model.set_adapter("default")

                feature_inputs = [
                    {**inp, "target": t} for inp, t in zip(inputs, outputs)
                ]

            query_labels = torch.tensor(query_labels).to(accelerator.device)

            feature_inputs = {
                k: v.to(accelerator.device)
                for k, v in collate_fn(feature_inputs).items()
            }

            with torch.inference_mode():
                feature_outputs = model(**feature_inputs, output_hidden_states=True)

                features = feature_outputs.hidden_states[-1][..., -1, :]

            [
                l.append(v)
                for l, v in zip(
                    (all_features, all_labels),
                    accelerator.gather_for_metrics((features, query_labels)),
                )
            ]

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        if accelerator.is_main_process:
            torch.save(
                {"features": all_features, "labels": all_labels},
                f"{log_dir}/outputs/{split_name}/features.pt",
            )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
