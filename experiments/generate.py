import os
import wandb
import pandas as pd
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import GenerationConfig

from llm.datasets import get_dataset, get_loader
from llm.models import get_model
from llm.models.peft import get_lora_model
from llm.logging import entrypoint

from llm.datasets.llm_utils_oe import prepare_oe_uncertainty_query

from llm.utils.generate_utils import generate_output


def generate_outputs_main(
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    num_workers=8,
    batch_size=1,
    model_name=None,
    model_dir=None,
    peft_dir=None,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    use_dataset_cache=True,
    prompt_style="oe",
    max_new_tokens=30,
    int8=False,
):
    accelerator = Accelerator()

    config = {
        "seed": seed,
        "model_name": model_name,
        "model_dir": model_dir,
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

    if peft_dir is not None:
        model = get_lora_model(
            model,
            peft_dir=peft_dir,
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
            num_workers=num_workers,
            use_cache=use_dataset_cache,
            prompt_style=prompt_style,
        )
        data_splits = list(filter(lambda x: x is not None, data_splits))
        data_split_names = ["train", "validation", "test"][: len(data_splits)]

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )

    for data, split in zip(data_splits, data_split_names):
        loader = get_loader(
            data,
            batch_size=batch_size,
            pin_memory=True,
            accelerator=accelerator,
        )

        output_generator = generate_output(
            accelerator,
            model,
            tokenizer,
            loader,
            generation_config=generation_config,
        )

        csv_path = f"{log_dir}/outputs/{split}"
        with accelerator.main_process_first():
            if accelerator.is_main_process:
                os.makedirs(csv_path)

        df = pd.DataFrame(output_generator)
        ## NOTE: Avoid spec errors when loading for labeling.
        df["query_label"] = -1

        df.to_csv(f"{csv_path}/{accelerator.process_index}.csv", index=False)


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

        _, query_labels, _ = prepare_oe_uncertainty_query(
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


def generate_labels_main(
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    num_workers=8,
    batch_size=1,
    model_name=None,
    model_dir=None,
    peft_dir=None,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    use_dataset_cache=True,
    strategy="substring",  ## fuzzy_gpt-3.5-turbo-1106
):
    accelerator = Accelerator()

    config = {
        "seed": seed,
        "model_name": model_name,
        "model_dir": model_dir,
        "peft_dir": peft_dir,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "log_dir": log_dir,
        "strategy": strategy,
    }
    if accelerator.is_main_process:
        wandb.config.update(config)

    tokenizer = get_model(
        f"{model_name}_tokenizer",
        model_dir=model_dir,
    )

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
            outputs=entrypoint(generate_outputs_main),
            labels=entrypoint(generate_labels_main),
        )
    )
