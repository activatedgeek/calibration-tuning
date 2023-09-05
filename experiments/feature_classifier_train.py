import torch
from tqdm import tqdm

import logging
import wandb
import pandas as pd
from accelerate import PartialState as AcceleratorState
from peft import PeftModel

from llm.logging import entrypoint
from llm.models import get_model, get_special_tokens
from llm.utils.evaluation import evaluate_dataset
from llm.utils.trainer import get_last_checkpoint_path
from llm.datasets import get_dataset, get_loader
from llm.utils.trainer import (
    TrainingArguments,
    ClassificationTrainer,
    WandbConfigUpdateCallback,
    get_last_checkpoint_path,
)

def prepare_model(
    causal_lm,
    accelerator,
    model_name,
    tokenizer,
    special_token_count,
    model_dir,
    peft_dir,
    fp8,
):
    model = get_model(
        model_name,
        device_map={"": accelerator.local_process_index},
        load_in_8bit=fp8,
        model_dir=model_dir,
        causal_lm=causal_lm,
    )

    model.resize_token_embeddings(len(tokenizer))
    if special_token_count:
        input_embeddings = model.get_input_embeddings().weight.data

        input_embeddings[-special_token_count:] = input_embeddings[
            :-special_token_count
        ].mean(dim=0, keepdim=True)

        if causal_lm:
            output_embeddings = model.get_output_embeddings().weight.data

            output_embeddings[-special_token_count:] = output_embeddings[
                :-special_token_count
            ].mean(dim=0, keepdim=True)

    # model = prepare_model_for_kbit_training(model)

    # if peft_dir is not None:
    #     peft_dir = get_last_checkpoint_path(peft_dir)

    #     model = PeftModel.from_pretrained(model, peft_dir)

    #     logging.info(f"Loaded PEFT checkpoint from '{peft_dir}'")

    return model

def main(
    seed=137,
    log_dir=None,
    dataset=None,
    kshot=0,
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
    adam_beta2=0.999,
    unc_decay=0.0,
    unc_decay_ratio=0.0,
    unc_normalize=True,
    weight_decay=0.0,
    loss_mode="reg",
    warmup_steps=100,
    max_steps=1000,
    save_steps=1000,
    logging_steps=25,
    use_dataset_cache=True,
):
    accelerator = AcceleratorState()

    config = {
        "seed": seed,
        "dataset": dataset,
        "model_name": model_name,
        "fp8": fp8,
        "model_dir": model_dir,
        "peft_dir": peft_dir,
        "eval_kshot": kshot,
    }
    if accelerator.is_main_process:
        wandb.config.update(config)

    tokenizer = get_model(
        f"{model_name}_tokenizer",
        model_dir=model_dir,
    )
    special_token_count = tokenizer.add_special_tokens(get_special_tokens(tokenizer))

    base_model = prepare_model(
        causal_lm=True,
        accelerator=accelerator,
        model_name=model_name,
        tokenizer=tokenizer,
        special_token_count=special_token_count,
        model_dir=model_dir,
        peft_dir=peft_dir,
        fp8=fp8,
    )
    model = prepare_model(
        causal_lm=False,
        accelerator=accelerator,
        model_name=model_name,
        tokenizer=tokenizer,
        special_token_count=special_token_count,
        model_dir=model_dir,
        peft_dir=peft_dir,
        fp8=fp8,
    )
    model.config.use_cache = False
    model.model = base_model.model #will have to change for peft

    for param in base_model.parameters():
        param.requires_grad = False

    with accelerator.main_process_first():
        train_data, val_data, test_data = get_dataset(
            dataset,
            root=data_dir,
            tokenizer=tokenizer,
            seed=seed,
            num_workers=num_workers,
            use_cache=use_dataset_cache,
            eval_kshot=kshot,
        )

    trainer = ClassificationTrainer(
        base_model=base_model,
        model=model,
        args=TrainingArguments(
            fsdp=False,
            fp16=not fp8,
            bf16=False,
            gradient_checkpointing=False,
            max_grad_norm=1.0,
            ddp_find_unused_parameters=False,
            max_steps=max_steps,
            eval_steps=1000,
            save_steps=save_steps,
            logging_steps=logging_steps,
            log_on_each_node=False,
            evaluation_strategy="steps",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            loss_mode=loss_mode,
            optim="sgd",
            # adam_beta1=0.9,
            # adam_beta2=adam_beta2,
            learning_rate=lr,
            lr_scheduler_type="constant",
            # warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            unc_decay=unc_decay,
            unc_decay_ratio=unc_decay_ratio,
            unc_normalize=unc_normalize,
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
