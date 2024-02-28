import json

import deepspeed
import torch
from torch import distributed as dist
from peft import prepare_model_for_kbit_training

from llm.logging import entrypoint
from llm.accelerate import AcceleratorState
from llm.models import get_model
from llm.models.peft import get_lora_model, get_temperature_scale_model
from llm.datasets import get_dataset
from llm.trainer import WandbConfigUpdateCallback, FineTuner


def main(
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    prompt_style=None,
    num_workers=4,
    batch_size=1,
    grad_acc=1,
    model_name=None,
    model_dir=None,
    peft_dir=None,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    lr=1e-4,
    weight_decay=0.0,
    scale_temp=False,
    warmup_steps=0,
    max_steps=1,
    log_steps=100,
    save_steps=1000,
    eval_steps=500,
    use_dataset_cache=True,
    resume_dir=None,
    int8=True,
    max_token_length=None,
    deepspeed_config_path=None,
    local_rank=None
):

    if deepspeed_config_path:
        deepspeed.init_distributed()
        local_rank = dist.get_rank()
        device_map = 'cuda'
        with open(deepspeed_config_path, 'rt') as f:
            deepspeed_config = json.load(f)
        if deepspeed_config and deepspeed_config['zero_optimization']['stage'] == 3:
            device_map = None
    else:
        accelerator = AcceleratorState()
        device_map = {"": accelerator.local_process_index}

    tokenizer = get_model(
        f"{model_name}_tokenizer",
        model_dir=model_dir,
    )

    model = get_model(
        model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        model_dir=model_dir,
        use_cache=False,
        tokenizer=tokenizer,
        load_in_8bit=int8,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    model = get_lora_model(
        model,
        peft_dir=peft_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        is_trainable=not scale_temp,
        adapter_name="default",
    )

    if scale_temp:
        model = get_temperature_scale_model(
            model,
            checkpoint_dir=peft_dir,
            is_trainable=True,
            weights_name=FineTuner.TEMPERATURE_WEIGHTS_NAME,
        )

    if deepspeed_config_path:
        dist.barrier()
        train_data, val_data, test_data = get_dataset(
            dataset,
            root=data_dir,
            tokenizer=tokenizer,
            seed=seed,
            num_workers=num_workers,
            use_cache=use_dataset_cache,
            prompt_style=prompt_style,
            max_token_length=max_token_length,
        )
    else:
        with accelerator.main_process_first():
            train_data, val_data, test_data = get_dataset(
                dataset,
                root=data_dir,
                tokenizer=tokenizer,
                seed=seed,
                num_workers=num_workers,
                use_cache=use_dataset_cache,
                prompt_style=prompt_style,
                max_token_length=max_token_length,
            )

    if scale_temp:
        train_data, val_data = val_data, test_data or val_data

    trainer = FineTuner(
        model=model,
        args=FineTuner.Args(
            seed=seed,
            fsdp=False,
            fp16=not torch.cuda.is_bf16_supported() and not model.is_loaded_in_8bit,
            bf16=torch.cuda.is_bf16_supported() and not model.is_loaded_in_8bit,
            gradient_checkpointing=False,
            ddp_find_unused_parameters=False,
            max_steps=max_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            logging_steps=log_steps,
            log_on_each_node=False,
            evaluation_strategy="steps",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            optim=('adamw_torch' if not deepspeed_config_path else transformers.TrainingArguments.default_optim),
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            gradient_accumulation_steps=grad_acc,
            output_dir=log_dir,
            report_to="wandb",
            dataloader_num_workers=num_workers,
            label_names=train_data.column_names,
            deepspeed=deepspeed_config if deepspeed_config_path else None,
            ## Custom.
            scale_temp=scale_temp,
        ),
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        callbacks=[
            WandbConfigUpdateCallback(
                dataset=dataset,
                data_dir=data_dir,
                model_name=model_name,
                model_dir=model_dir,
                peft_dir=peft_dir,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                prompt_style=prompt_style,
            ),
        ],
    )
    trainer.train(resume_from_checkpoint=resume_dir)
    trainer.save_state()


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint(main))
