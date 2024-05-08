import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from llm.logging import entrypoint
from llm.datasets import register_dataset, get_dataset, get_loader
from llm.models.peft.temperature_scaling import TemperatureScale


class LogitsDataset(Dataset):
    def __init__(self, logits, labels):
        self.logits = logits
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.logits[index], self.labels[index]


@register_dataset(attrs=dict(unlisted=True))
def offline_logits(root=None, **_):
    data_dir = Path(root) / "offline_logits"

    data_splits = dict()
    for split in ["validation", "test"]:
        data_path = data_dir / split
        data_path = next(data_path.glob("*.bin"))

        data = torch.load(data_path, map_location="cpu")

        logits = data.pop("logits")
        labels = data.pop("labels")

        data_splits[split] = LogitsDataset(logits, labels)

    val_data = data_splits.pop("validation")
    test_data = data_splits.pop("test")

    return None, val_data, test_data


@entrypoint(with_accelerator=True)
def main(
    accelerator=None,
    seed=137,
    log_dir=None,
    dataset="offline_logits",
    data_dir=None,
    num_workers=4,
    batch_size=64,
    lr=1e-2,
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
    )

    model, optimizer, scheduler, loader = accelerator.prepare(
        model, optimizer, scheduler, loader
    )

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
