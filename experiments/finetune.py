import torch
from accelerate import PartialState as AcceleratorState

from llm.datasets import get_dataset
from llm.datasets.llm_utils import tokenize_datasets
from llm.models import get_model
from llm.models.peft import get_lora_model
from llm.models.peft.temperature_scaling import (
    get_temperature_scaled_model,
    TemperatureSaveCallback,
)
from llm.logging import entrypoint
from llm.utils.trainer import (
    TrainingArguments,
    UncertaintyTrainer,
    WandbConfigUpdateCallback,
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
    label_smoothing=0.0,
    weight_decay=0.0,
    loss_mode="reg",
    scale_temp=False,
    warmup_steps=0,
    max_steps=1,
    log_steps=100,
    save_steps=1000,
    eval_steps=1000,
    use_dataset_cache=True,
    resume_dir=None,
):
    accelerator = AcceleratorState()

    tokenizer = get_model(
        f"{model_name}_tokenizer",
        model_dir=model_dir,
    )

    model = get_model(
        model_name,
        device_map={"": accelerator.local_process_index},
        torch_dtype=torch.float16,
        model_dir=model_dir,
        use_cache=False,
        tokenizer=tokenizer,
        load_in_8bit=fp8,
    )

    ## If resuming, resume_dir takes priority. Otherwise, to avoid clashes with saved state.
    peft_dir = resume_dir or peft_dir
    model = get_lora_model(
        model,
        peft_dir=peft_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    model = get_temperature_scaled_model(model, checkpoint_dir=peft_dir, is_trainable=scale_temp)

    with accelerator.main_process_first():
        train_data, val_data, test_data = get_dataset(
            dataset,
            root=data_dir,
            tokenizer=tokenizer,
            seed=seed,
            num_workers=num_workers,
            use_cache=use_dataset_cache,
        )

        train_data, val_data, test_data = tokenize_datasets(
            tokenizer, train_data, val_data, test_data
        )

    trainer = UncertaintyTrainer(
        model=model,
        args=TrainingArguments(
            seed=seed,
            fsdp=False,
            fp16=not fp8,
            bf16=False,
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
            label_smoothing=label_smoothing,
        ),
        train_dataset=train_data,
        eval_dataset=val_data,
        test_dataset=test_data,
        tokenizer=tokenizer,
        callbacks=[
            WandbConfigUpdateCallback(
                dataset=dataset,
                data_dir=data_dir,
                model_name=model_name,
                model_dir=model_dir,
                peft_dir=peft_dir,
                resume_dir=resume_dir,
                fp8=fp8,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            ),
            TemperatureSaveCallback(),
        ],
    )
    trainer.train(resume_from_checkpoint=resume_dir)
    trainer.save_state()


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint(main))
