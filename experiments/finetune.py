import logging
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
import transformers
from transformers import TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training

from llm.logging import set_logging
from llm.datasets import get_dataset, get_num_workers
from llm.models import create_model, get_special_tokens
from llm.utils import CalibrationTrainer


@dataclass
class ArgsTrain:
    seed: int = field(default=None)
    log_dir: str = field(default=None)
    data_dir: str = field(default=None)
    dataset: str = field(default=None)
    dataset_instance: str = field(default=None)
    batch_size: int = field(default=1)
    lr: float = field(default=5e-2)
    unc_decay: float = field(default=1.0)
    weight_decay: float = field(default=2e-5)
    warmup_steps: int = field(default=0)
    epochs: int = field(default=0)


@dataclass
class ArgsModel:
    model_dir: str = field(default=None)
    model_name: str = field(default=None)
    fp8: bool = field(default=True)
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)


def main(
    accelerator,
    seed=None,
    log_dir=None,
    data_dir=None,
    model_dir=None,
    dataset=None,
    dataset_instance=None,
    batch_size=1,
    model_name=None,
    fp8=True,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    lr=5e-2,
    unc_decay=1.0,
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
        instance=dataset_instance,
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

    if special_token_count:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings[-special_token_count:] = input_embeddings[
            :-special_token_count
        ].mean(dim=0, keepdim=True)
        output_embeddings[-special_token_count:] = output_embeddings[
            :-special_token_count
        ].mean(dim=0, keepdim=True)

    model = prepare_model_for_int8_training(model)
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
        save_steps=1000, ## FIXME: saving leads to OOM.
        logging_steps=100,
        evaluation_strategy="steps",
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
    trainer = CalibrationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        test_dataset=test_data,
        tokenizer=tokenizer,
        unc_decay=unc_decay,
    )
    trainer.train()
    trainer.save_state()


def entrypoint():
    parser = transformers.HfArgumentParser((ArgsModel, ArgsTrain))
    model_args, train_args = parser.parse_args_into_dataclasses()

    kwargs = dict(**asdict(model_args), **asdict(train_args))

    set_logging(log_dir=train_args.log_dir, use_wandb=False)

    accelerator = Accelerator()
    if accelerator.is_main_process:
        logging.info(f"Working with {accelerator.num_processes} process(es).")

    main(accelerator, **kwargs)


if __name__ == "__main__":
    entrypoint()
