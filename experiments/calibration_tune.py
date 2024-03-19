from llm.datasets import get_dataset
from llm.distributed import AcceleratorState
from llm.logging import entrypoint
from llm.models import get_model
from llm.models.peft import get_lora_model, get_temperature_head
from llm.trainer import WandbConfigUpdateCallback, CalibrationTuner


@entrypoint
def main(
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    prompt_style=None,
    max_token_length=None,
    num_workers=4,
    use_dataset_cache=True,
    model_name=None,
    int8=True,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    peft_dir=None,
    ref_peft_dir=None,
    scale_temp=False,
    batch_size=1,
    lr=1e-4,
    warmup_ratio=0.0,
    kl_decay=0.0,
    max_steps=1,
):
    accelerator = AcceleratorState()

    trainer_args = CalibrationTuner.Args(
        seed=seed,
        output_dir=log_dir,
        max_steps=max_steps,
        eval_steps=max_steps // 10,
        save_steps=max_steps // 10,
        logging_steps=max(1, max_steps // 200),
        dataloader_num_workers=num_workers,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        scale_temp=scale_temp,
        kl_decay=kl_decay,
    )

    with accelerator.main_process_first():
        train_data, val_data, test_data = get_dataset(
            dataset,
            root=data_dir,
            seed=seed,
            prompt_style=prompt_style,
            max_token_length=max_token_length,
            num_workers=num_workers,
            use_cache=use_dataset_cache,
        )
    if scale_temp:
        train_data, val_data = val_data, test_data or val_data

    tokenizer, model = get_model(
        model_name,
        device_map={"": accelerator.local_process_index},
        use_int8=int8,
    )

    model = get_lora_model(
        model,
        peft_id_or_dir=peft_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        is_trainable=not scale_temp,
        adapter_name="default",
    )

    model = get_lora_model(
        model,
        peft_id_or_dir=ref_peft_dir or peft_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        is_trainable=False,
        adapter_name="_ref",
    )

    if scale_temp:
        model.requires_grad_(False)

        temperature_model = get_temperature_head(
            checkpoint_dir=peft_dir,
            is_trainable=True,
            weights_name=CalibrationTuner.TEMPERATURE_WEIGHTS_NAME,
        ).to(accelerator.local_process_index)

        ## HOTFIX: To allow registry with Trainer optimizer.
        model.temperature_model = temperature_model
    else:
        temperature_model = None

    trainer = CalibrationTuner(
        model=model,
        query_temperature_model=temperature_model,
        args=trainer_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        callbacks=[
            WandbConfigUpdateCallback(
                dataset=dataset,
                data_dir=data_dir,
                prompt_style=prompt_style,
                max_token_length=max_token_length,
                model_name=model_name,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                peft_dir=peft_dir,
            ),
        ],
    )
    trainer.train()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
