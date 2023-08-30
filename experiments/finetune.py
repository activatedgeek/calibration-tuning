from accelerate import PartialState as AcceleratorState
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from llm.datasets import get_dataset
from llm.models import get_model, get_special_tokens
from llm.logging import entrypoint
from llm.utils.trainer import (
    TrainingArguments,
    CalibrationTrainer,
    WandbConfigUpdateCallback,
    get_last_checkpoint_path,
)


def main(
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    num_workers=8,
    batch_size=1,
    grad_acc=1,
    model_name=None,
    model_dir=None,
    peft_dir=None,
    fp8=True,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    lr=1e-4,
    unc_decay=0.0,
    weight_decay=0.0,
    loss_mode="reg",
    warmup_steps=100,
    max_steps=1000,
    save_steps=1000,
    use_dataset_cache=True,
):
    accelerator = AcceleratorState()

    tokenizer = get_model(
        f"{model_name}_tokenizer",
        model_dir=model_dir,
    )
    special_token_count = tokenizer.add_special_tokens(get_special_tokens(tokenizer))

    model = get_model(
        model_name,
        device_map={"": accelerator.local_process_index},
        load_in_8bit=fp8,
        model_dir=model_dir,
    )
    model.config.use_cache = False

    ## NOTE: Token embeddings aren't trained.
    model.resize_token_embeddings(len(tokenizer))
    if special_token_count:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings[-special_token_count:] = input_embeddings[
            :-special_token_count
        ].mean(dim=0, keepdim=True)
        output_embeddings[-special_token_count:] = output_embeddings[
            :-special_token_count
        ].mean(dim=0, keepdim=True)

    model = prepare_model_for_kbit_training(model)

    if peft_dir is not None:
        peft_dir = get_last_checkpoint_path(peft_dir)

        model = PeftModel.from_pretrained(model, peft_dir, is_trainable=True)

        if accelerator.is_main_process:
            print(f"[INFO]: Loaded PEFT checkpoint from '{peft_dir}'")
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        model = get_peft_model(model, peft_config)

    with accelerator.main_process_first():
        train_data, val_data, test_data = get_dataset(
            dataset,
            root=data_dir,
            tokenizer=tokenizer,
            seed=seed,
            num_workers=num_workers,
            use_cache=use_dataset_cache,
        )

    trainer = CalibrationTrainer(
        model=model,
        args=TrainingArguments(
            fsdp=False,
            fp16=not fp8,
            bf16=False,
            gradient_checkpointing=False,
            ddp_find_unused_parameters=False,
            max_steps=max_steps,
            eval_steps=1000,
            save_steps=save_steps,
            logging_steps=100,
            log_on_each_node=False,
            evaluation_strategy="steps",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            loss_mode=loss_mode,
            optim="adamw_torch",
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            unc_decay=unc_decay,
            gradient_accumulation_steps=grad_acc,
            output_dir=log_dir,
            report_to="wandb",
            dataloader_num_workers=4,
        ),
        train_dataset=train_data,
        eval_dataset=val_data,
        test_dataset=test_data,
        tokenizer=tokenizer,
        callbacks=[
            WandbConfigUpdateCallback(
                seed=seed,
                dataset=dataset,
                data_dir=data_dir,
                model_name=model_name,
                model_dir=model_dir,
                peft_dir=peft_dir,
                fp8=fp8,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            ),
        ],
    )
    trainer.train(resume_from_checkpoint=peft_dir)
    trainer.save_state()


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint(main))
