import logging
from pathlib import Path
import torch
from tqdm.auto import tqdm

from llm.logging import entrypoint
from llm.datasets import get_dataset, get_loader
from llm.models.peft.temperature_scaling import TemperatureScale


@entrypoint(with_accelerator=True)
def main(
    accelerator=None,
    seed=137,
    log_dir=None,
    dataset="offline_logits",
    data_dir=None,
    num_workers=4,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-2,
    max_steps=2000,
):
    _, val_data, test_data = get_dataset(dataset, root=data_dir)
    if val_data is None:
        val_data = test_data

    model = TemperatureScale()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    loader = get_loader(
        val_data,
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
        optimizer.zero_grad()

        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)

        logits, labels = batch
        logits = model(logits)

        loss = criterion(logits, labels)

        accelerator.backward(loss)

        optimizer.step()
        scheduler.step()

        log_metrics = {
            "loss": loss.detach().item(),
            "log_temperature": model.log_temperature.data.item(),
            "temperature": model.log_temperature.data.exp().item(),
        }

        if accelerator.is_main_process and (step + 1) % logging_steps == 0:
            logging.info(log_metrics)
            logging.info(log_metrics, extra=dict(metrics=True))

        if accelerator.is_main_process and (step + 1) % save_steps == 0:
            checkpoint_path = (
                Path(log_dir) / f"checkpoint-{step + 1}" / "temperature_head.bin"
            )
            checkpoint_path.parent.mkdir()

            torch.save(accelerator.unwrap_model(model).state_dict(), checkpoint_path)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
