import os
from dataclasses import dataclass, field, asdict
import transformers
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training

from llm.datasets import get_dataset
from llm.models import get_model, get_special_tokens
from llm.utils import TrainingArguments, CalibrationTrainer


@dataclass
class ArgsData:
    data_dir: str = field(default=None)
    dataset: str = field(default=None)
    dataset_instance: str = field(default=None)
    num_workers: int = field(default=8)
    eval_kshot: int = field(default=5)


@dataclass
class ArgsModel:
    model_dir: str = field(default=None)
    model_name: str = field(default=None)
    fp8: bool = field(default=True)
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)


@dataclass
class ArgsTrain:
    log_dir: str = field(default=None)
    seed: int = field(default=137)
    batch_size: int = field(default=1)
    grad_acc: int = field(default=1)
    lr: float = field(default=3e-5)
    weight_decay: float = field(default=1e-6)
    unc_decay: float = field(default=0.0)
    unc_normalize: bool = field(default=True)
    loss_mode: str = field(default="reg")
    warmup_steps: int = field(default=100)
    epochs: int = field(default=1)


def main(
    seed=None,
    log_dir=None,
    dataset=None,
    dataset_instance=None,
    data_dir=None,
    eval_kshot=5,
    num_workers=8,
    batch_size=1,
    grad_acc=1,
    model_name=None,
    model_dir=None,
    fp8=True,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    lr=1e-4,
    unc_decay=0.0,
    unc_normalize=True,
    weight_decay=0.0,
    loss_mode="reg",
    warmup_steps=100,
    epochs=1,
):
    training_args = TrainingArguments(
        fsdp=False,
        fp16=not fp8,
        bf16=False,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        num_train_epochs=epochs,
        eval_steps=1000,
        save_steps=1000,
        logging_steps=100,
        log_on_each_node=False,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        loss_mode=loss_mode,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        unc_decay=unc_decay,
        unc_normalize=unc_normalize,
        gradient_accumulation_steps=grad_acc,
        output_dir=log_dir,
        report_to="wandb",
        dataloader_num_workers=4,
    )

    tokenizer = get_model(
        f"{model_name}_tokenizer",
        model_dir=model_dir,
    )
    special_token_count = tokenizer.add_special_tokens(get_special_tokens(tokenizer))

    train_data, val_data, test_data = get_dataset(
        dataset,
        instance=dataset_instance,
        eval_kshot=eval_kshot,
        root=data_dir,
        tokenizer=tokenizer,
        seed=seed,
        num_workers=num_workers,
    )

    model = get_model(
        model_name,
        device_map={"": training_args.local_rank},
        load_in_8bit=fp8,
        model_dir=model_dir,
    )

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

    model = prepare_model_for_int8_training(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(model, peft_config)

    trainer = CalibrationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        test_dataset=test_data,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_state()


def entrypoint():
    parser = transformers.HfArgumentParser((ArgsData, ArgsModel, ArgsTrain))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()
    train_args.log_dir = train_args.log_dir or os.environ.get("WANDB_DIR")

    main(**asdict(data_args), **asdict(model_args), **asdict(train_args))


if __name__ == "__main__":
    entrypoint()
