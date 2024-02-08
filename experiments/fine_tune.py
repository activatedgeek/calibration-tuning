import torch
from peft import prepare_model_for_kbit_training

from llm.logging import entrypoint
from llm.accelerate import AcceleratorState
from llm.models import get_model
from llm.models.peft import get_lora_model, get_temperature_scaled_model
from llm.datasets import get_dataset
from llm.trainer import WandbConfigUpdateCallback, FineTuner


def main(
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    prompt_style="choice",
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
    eval_steps=1000,
    use_dataset_cache=True,
    resume_dir=None,
    int8=True,
):
    accelerator = AcceleratorState()

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
        model = get_temperature_scaled_model(
            model, peft_dir=peft_dir, is_trainable=True
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
            optim="adamw_torch",
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            gradient_accumulation_steps=grad_acc,
            output_dir=log_dir,
            report_to="wandb",
            dataloader_num_workers=num_workers,
            label_names=train_data.column_names,
        ),
        train_dataset=train_data,
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
                scale_temp=scale_temp,
            ),
        ],
    )
    trainer.train(resume_from_checkpoint=resume_dir)
    trainer.save_state()


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint(main))
