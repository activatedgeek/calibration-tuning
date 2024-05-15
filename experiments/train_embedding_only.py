import logging
from collections import OrderedDict
from pathlib import Path
import torch
from tqdm.auto import tqdm
import wandb

from llm.logging import entrypoint
from llm.datasets import get_dataset, get_loader
from llm.models.peft import get_classifier_head, get_temperature_head
from llm.trainer import ClassificationTuner


@torch.inference_mode
def compute_metrics(accelerator, data, model, batch_size=64, num_workers=8, prefix=""):
    model.eval()

    loader = get_loader(
        data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        accelerator=accelerator,
    )

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    all_data = OrderedDict([("loss", []), ("acc", [])])

    for inputs in tqdm(loader, leave=False):
        embeddings = inputs.get("embedding")
        labels = inputs.get("query_label")

        logits = model(embeddings)

        loss = criterion(logits, labels)

        preds = labels == logits.argmax(dim=-1)

        [
            all_data[k].append(v.cpu())
            for k, v in zip(
                all_data.keys(), accelerator.gather_for_metrics((loss, preds))
            )
        ]

    all_data = {
        f"{prefix}{k}": torch.cat(v, dim=0).float().mean().item()
        for k, v in all_data.items()
    }

    return all_data


@entrypoint(with_accelerator=True)
def main(
    accelerator=None,
    seed=137,
    log_dir=None,
    dataset=None,
    prompt_style=None,
    data_dir=None,
    num_workers=4,
    model_dir=None,
    scale_temp=False,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-2,
    max_steps=2000,
):
    config = dict(
        seed=seed,
        log_dir=log_dir,
        dataset=dataset,
        prompt_style=prompt_style,
        model_dir=model_dir,
        scale_temp=scale_temp,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        max_steps=max_steps,
    )
    if accelerator.is_main_process:
        wandb.config.update(config)

    train_data, val_data, test_data = get_dataset(
        dataset,
        root=data_dir,
        seed=seed,
        prompt_style=prompt_style,
        num_workers=num_workers,
    )
    if scale_temp:
        train_data, val_data = val_data, test_data

    model = get_classifier_head(
        input_size=train_data[0]["embedding"].shape[0],
        checkpoint_dir=model_dir,
        is_trainable=not scale_temp,
        weights_name=ClassificationTuner.WEIGHTS_NAME,
    )

    if scale_temp:
        temperature_model = get_temperature_head(is_trainable=True)

        model = torch.nn.Sequential(
            model,
            temperature_model,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    loader = get_loader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        accelerator=accelerator,
        shuffle=True,
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    criterion = torch.nn.CrossEntropyLoss()

    logging_steps = max(1, max_steps // 200)
    save_steps = max_steps // 10

    iter_loader = iter(loader)
    for step in tqdm(range(max_steps)):
        model.train()

        optimizer.zero_grad()

        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)

        embeddings = batch.get("embedding")
        labels = batch.get("query_label")

        logits = model(embeddings)

        loss = criterion(logits, labels)

        accelerator.backward(loss)

        optimizer.step()
        scheduler.step()

        train_metrics = {
            "train/loss": loss.detach().item(),
        }

        if (step + 1) % logging_steps == 0:
            if val_data is not None:
                val_metrics = compute_metrics(
                    accelerator,
                    val_data,
                    model,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    prefix="eval/",
                )
                logging.info(val_metrics, extra=dict(metrics=True))
                logging.debug(val_metrics)

            logging.info(train_metrics, extra=dict(metrics=True))
            logging.debug(train_metrics)

        if accelerator.is_main_process and (step + 1) % save_steps == 0:
            checkpoint_path = (
                Path(log_dir)
                / f"checkpoint-{step + 1}"
                / ClassificationTuner.WEIGHTS_NAME
            )
            checkpoint_path.parent.mkdir()

            torch.save(accelerator.unwrap_model(model).state_dict(), checkpoint_path)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
