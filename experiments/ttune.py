import logging
from tqdm.auto import tqdm
import wandb
import torch
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup

from llm.datasets import get_dataset, get_loader
from llm.datasets.llm_utils import tokenize_datasets, DataCollatorForSupervisedDataset
from llm.models import get_model
from llm.models.peft import get_temperature_scaled_model, save_temperature_scaled_model
from llm.logging import entrypoint


def main(
    seed=137,
    log_dir=None,
    model_name=None,
    model_dir=None,
    checkpoint_dir=None,
    fp8=True,
    dataset=None,
    data_dir=None,
    use_dataset_cache=True,
    batch_size=1,
    grad_acc=1,
    lr=1e-3,
    warmup_steps=0,
    max_steps=0,
    log_steps=100,
    save_steps=1000,
):
    accelerator = Accelerator(gradient_accumulation_steps=grad_acc)

    if accelerator.is_main_process:
        wandb.config.update(
            {
                "seed": seed,
                "log_dir": log_dir,
                "model_name": model_name,
                "model_dir": model_dir,
                "checkpoint_dir": checkpoint_dir,
                "fp8": fp8,
                "dataset": dataset,
                "data_dir": data_dir,
                "batch_size": batch_size,
                "grad_acc": grad_acc,
                "lr": lr,
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "log_steps": log_steps,
                "save_steps": save_steps,
            }
        )

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
    model = get_temperature_scaled_model(
        model,
        module_name="lm_head",
        checkpoint_dir=checkpoint_dir,
    )

    with accelerator.main_process_first():
        train_data, _, _ = get_dataset(
            dataset,
            root=data_dir,
            tokenizer=tokenizer,
            seed=seed,
            use_cache=use_dataset_cache,
        )

        (train_data,) = tokenize_datasets(tokenizer, train_data)

    train_loader = get_loader(
        train_data,
        batch_size=batch_size,
        accelerator=accelerator,
        pin_memory=True,
        shuffle=True,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    train_iter = train_loader.__iter__()
    for s in tqdm(range(1, max_steps + 1)):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = train_loader.__iter__()
            batch = next(train_iter)

        with accelerator.accumulate(model):
            model.train()
            optimizer.zero_grad()

            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss

            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()

        if s % log_steps == 0:
            temperature = (
                accelerator.unwrap_model(model)
                .lm_head[-1]
                .log_temperature.detach()
                .exp()
            )
            logging.info(
                {"loss": loss.detach().float(), "temperature": temperature},
                extra=dict(metrics=True),
            )
            logging.debug({"loss": loss.detach().float(), "temperature": temperature})

        if s % save_steps == 0:
            if accelerator.is_main_process:
                save_temperature_scaled_model(
                    accelerator.unwrap_model(model), f"{log_dir}/checkpoint-{s}"
                )


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint(main))
