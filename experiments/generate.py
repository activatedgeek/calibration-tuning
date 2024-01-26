import os
import wandb
import pandas as pd
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from peft import PeftModel
from transformers import GenerationConfig

from llm.datasets import get_dataset, get_loader
from llm.datasets.llm_utils import LMText
from llm.models import get_model
from llm.models.peft import get_lora_model
from llm.logging import entrypoint

from llm.datasets.llm_utils_oe import prepare_oe_calibration_query

from llm.utils.generate_utils import generate_output


def prepare_model(
    accelerator,
    model_name=None,
    model_dir=None,
    peft_dir=None,
    lora_rank=None,
    lora_alpha=None,
    lora_dropout=None,
):
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

    return tokenizer, model


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
):
    accelerator = Accelerator()

    assert (
        batch_size == 1
    ), "Only use batch size 1 for now to avoid left padding issues."

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
    }
    if accelerator.is_main_process:
        wandb.config.update(config)

    tokenizer, model = prepare_model(
        accelerator,
        model_name=model_name,
        model_dir=model_dir,
        peft_dir=peft_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    with accelerator.main_process_first():
        train_data, _, _ = get_dataset(
            dataset,
            root=data_dir,
            tokenizer=tokenizer,
            seed=seed,
            num_workers=num_workers,
            use_cache=use_dataset_cache,
            prompt_style=prompt_style,
        )

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )

    for data, split in zip([train_data], ["train"]):
        loader = get_loader(
            data,
            batch_size=batch_size,
            pin_memory=True,
            accelerator=accelerator,
            collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0].keys()},
        )

        output_generator = generate_output(
            accelerator,
            model,
            tokenizer,
            loader,
            prompt_style=prompt_style,
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
    model,
    tokenizer,
    loader,
    query_format="roman_choice",
    comparison_strategy="substring",
):
    if isinstance(model, PeftModel):
        model.set_adapter("default")

    for inputs in tqdm(loader):
        inputs_list = []
        for i in range(len(inputs[next(iter(inputs))])):
            new_dict = {key: value[i] for key, value in inputs.items()}
            inputs_list.append(new_dict)

        question_strings = []
        for x in inputs_list:
            x.pop("target")
            question_strings.append(str(LMText.from_(x)))

        output_strings = inputs["output"]
        oe_target_strings = inputs["target"]
        # Find the rightmost occurrence of the eos token.
        for i, x in enumerate(oe_target_strings):
            index = x.rfind(tokenizer.eos_token)
            if index != -1:
                # Everything before the substring + everything after the substring
                oe_target_strings[i] = x[:index]

        _, acc = prepare_oe_calibration_query(
            tokenizer,
            oe_target_strings,
            output_strings,
            question_strings,
            format=query_format,
            comparison_strategy=comparison_strategy,
        )

        outputs = [
            LMText(**{**dict(zip(inputs.keys(), vals)), "target": ""})
            for vals in zip(*inputs.values())
        ]
        outputs = [
            {
                **s.to_pydict(),
                "target": inputs["target"][i],
                "query_label": int(t.item()),
            }
            for i, (s, t) in enumerate(zip(outputs, acc))
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

    tokenizer, model = prepare_model(
        accelerator,
        model_name=model_name,
        model_dir=model_dir,
        peft_dir=peft_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    with accelerator.main_process_first():
        train_data, _, _ = get_dataset(
            dataset,
            root=data_dir,
            tokenizer=tokenizer,
            seed=seed,
            num_workers=num_workers,
            use_cache=use_dataset_cache,
        )

    for data, split in zip([train_data], ["train"]):

        loader = get_loader(
            data,
            batch_size=batch_size,
            pin_memory=True,
            accelerator=accelerator,
            collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0].keys()},
        )

        label_generator = generate_query_label(
            accelerator,
            model,
            tokenizer,
            loader,
            comparison_strategy=strategy,
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
