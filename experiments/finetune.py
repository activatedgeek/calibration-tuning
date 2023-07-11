import logging
from accelerate import Accelerator
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training

from llm.logging import set_logging
from llm.datasets import get_dataset, get_num_workers
from llm.models import create_model, get_special_tokens


def main(
    accelerator,
    seed=None,
    log_dir=None,
    data_dir=None,
    model_dir=None,
    dataset=None,
    batch_size=1,
    model_name=None,
    fp8=True,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    lr=5e-2,
    weight_decay=2e-5,
    warmup_steps=0,
    epochs=0,
):
    tokenizer = create_model(
        model_name=f"{model_name}_tokenizer", model_kwargs=dict(cache_dir=model_dir)
    )
    special_token_count = tokenizer.add_special_tokens(get_special_tokens(tokenizer))

    train_data, val_data, test_data = get_dataset(
        dataset,
        root=data_dir,
        tokenizer=tokenizer,
        seed=seed,
    )

    model = create_model(
        model_name=model_name,
        model_kwargs=dict(
            device_map={"": accelerator.device},
            load_in_8bit=fp8,
            cache_dir=model_dir,
        ),
    )
    model.resize_token_embeddings(len(tokenizer))
    ## Mean init new embeddings.
    if special_token_count:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings[-special_token_count:] = input_embeddings[
            :-special_token_count
        ].mean(dim=0, keepdim=True)
        output_embeddings[-special_token_count:] = output_embeddings[
            :-special_token_count
        ].mean(dim=0, keepdim=True)

    if fp8:
        model = prepare_model_for_int8_training(model)
    if lora_rank:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        local_rank=accelerator.local_process_index,
        fsdp=False,
        fp16=not fp8,
        bf16=False,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        num_train_epochs=epochs,
        eval_steps=1000,
        save_steps=1000,
        logging_steps=10,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        optim="adamw_torch",
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        gradient_accumulation_steps=1,
        output_dir=log_dir,
        report_to="wandb",
        dataloader_num_workers=get_num_workers(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
    )
    trainer.train()
    if accelerator.is_main_process:
        trainer.save_state()
        trainer.save_model(output_dir=log_dir)


def entrypoint(seed=None, log_dir=None, **kwargs):
    accelerator = Accelerator()

    ## Only setup logging from one process.
    log_dir, finish_logging = (
        set_logging(log_dir=log_dir) if accelerator.is_main_process else [None, None]
    )
    if accelerator.is_main_process:
        logging.info(f"Working with {accelerator.num_processes} process(es).")

    # with FixedSeedAll(seed):
    main(accelerator, **kwargs, seed=seed, log_dir=log_dir)

    if accelerator.is_main_process:
        finish_logging()


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint)
