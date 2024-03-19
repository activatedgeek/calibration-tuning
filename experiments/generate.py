import os
import wandb
import pandas as pd
from tqdm.auto import tqdm
from transformers import GenerationConfig

import multiprocess.context as ctx

## @HOTFIX: for hanging processes in dataset map.
ctx._force_start_method("spawn")

from llm.datasets import get_dataset, get_loader, prepare_uncertainty_query
from llm.logging import entrypoint_with_accelerator
from llm.models import get_model
from llm.models.peft import get_lora_model
from llm.utils.generate_utils import generate_output


@entrypoint_with_accelerator
def generate_outputs_main(
    accelerator=None,
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    batch_size=1,
    model_name=None,
    peft_dir=None,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    use_dataset_cache=True,
    prompt_style=None,
    kshot=0,
    max_new_tokens=30,
    int8=False,
):
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

    if peft_dir is not None:
        model = get_lora_model(
            model,
            peft_id_or_dir=peft_dir,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            is_trainable=False,
            adapter_name="default",
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

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )

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

        generate_output(
            accelerator,
            model,
            tokenizer,
            loader,
            generation_config=generation_config,
            log_dir=f"{log_dir}/outputs/{split_name}",
        )


def generate_query_label(
    accelerator,
    tokenizer,
    loader,
    query_format="roman_choice",
    strategy="substring",
):
    for inputs in tqdm(loader):
        inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
        targets = [inp.pop("target") for inp in inputs]
        outputs = [inp.pop("output") for inp in inputs]

        _, query_labels, _ = prepare_uncertainty_query(
            tokenizer,
            inputs,
            targets,
            outputs,
            format=query_format,
            strategy=strategy,
        )

        outputs = [
            {**inp, "target": tgt, "output": out, "query_label": int(ql.item())}
            for inp, tgt, out, ql in zip(inputs, targets, outputs, query_labels)
        ]

        yield from outputs


@entrypoint_with_accelerator
def generate_labels_main(
    accelerator=None,
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    num_workers=8,
    batch_size=1,
    model_name=None,
    peft_dir=None,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    use_dataset_cache=True,
    strategy="substring",  ## fuzzy_gpt-3.5-turbo-1106
):
    config = {
        "seed": seed,
        "model_name": model_name,
        "peft_dir": peft_dir,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "log_dir": log_dir,
        "strategy": strategy,
    }
    if accelerator.is_main_process:
        wandb.config.update(config)

    tokenizer = get_model(f"{model_name}_tokenizer")

    with accelerator.main_process_first():
        data_splits = get_dataset(
            dataset,
            root=data_dir,
            tokenizer=tokenizer,
            seed=seed,
            num_workers=num_workers,
            use_cache=use_dataset_cache,
        )
        data_splits = list(filter(lambda x: x is not None, data_splits))
        data_split_names = ["train", "validation", "test"][: len(data_splits)]

    for data, split in zip(data_splits, data_split_names):
        loader = get_loader(
            data,
            batch_size=batch_size,
            pin_memory=True,
            accelerator=accelerator,
        )

        label_generator = generate_query_label(
            accelerator,
            tokenizer,
            loader,
            strategy=strategy,
        )

        csv_path = f"{log_dir}/labels/{split}"
        with accelerator.main_process_first():
            if accelerator.is_main_process:
                os.makedirs(csv_path)

        pd.DataFrame(label_generator).to_csv(
            f"{csv_path}/{accelerator.process_index}.csv", index=False
        )


if __name__ == "__main__":
    import fire

    fire.Fire(
        dict(
            outputs=generate_outputs_main,
            labels=generate_labels_main,
        )
    )
