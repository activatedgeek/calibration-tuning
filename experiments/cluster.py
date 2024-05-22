import logging
from collections import OrderedDict
from tqdm.auto import tqdm
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from llm.datasets import get_dataset_attrs, get_dataset, get_loader
from llm.logging import entrypoint


class CategoryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


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
        embeddings, labels = inputs

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
    data_dir=None,
    prompt_style="oe",
    use_dataset_cache=True,
    model_name="mistral-7b_instruct",
    batch_size=64,
    lr=1e-2,
    weight_decay=1e-1,
    max_steps=100,
):
    config = dict(
        seed=seed,
        log_dir=log_dir,
        prompt_style=prompt_style,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        max_steps=max_steps,
    )
    if accelerator.is_main_process:
        wandb.config.update(config)

    supercategories = get_dataset_attrs("mmlu").get("task_categories")
    slabels = {v: idx for idx, v in enumerate(set(supercategories.values()))}

    all_data = {"validation": [], "test": []}

    all_datasets = get_dataset(f"mmlu_offline_all:{model_name}")
    for dataset in tqdm(all_datasets, leave=False):
        with accelerator.main_process_first():
            data_splits = get_dataset(
                dataset,
                root=data_dir,
                seed=seed,
                use_cache=use_dataset_cache,
                prompt_style=prompt_style,
            )

        sc = supercategories[dataset.split(":")[-1]]

        data_splits = [
            (
                s,
                ds.select(
                    np.random.default_rng(seed=seed).choice(
                        len(ds),
                        len(ds) if s == "validation" else min(len(ds), 10),
                        replace=False,
                    )
                ),
            )
            for s, ds in zip(["train", "validation", "test"], data_splits)
            if ds is not None
        ]

        for split, data in data_splits:
            for d in data:
                label = slabels[sc]
                # if split != "validation":
                #     label = np.random.randint(0, 4)
                all_data[split].append((d["embedding"], label))

    all_data = {k: CategoryDataset(v) for k, v in all_data.items()}

    train_data = all_data.pop("test")
    val_data = all_data.pop("validation")

    logging.info(
        f"Collected train ({len(train_data)}) / validation (N = {len(val_data)})"
    )

    model = nn.Linear(1536, 4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    loader = get_loader(
        train_data,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        accelerator=accelerator,
        shuffle=True,
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    criterion = torch.nn.CrossEntropyLoss()

    logging_steps = max(1, max_steps // 10)
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

        embeddings, labels = batch

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
                    num_workers=4,
                    prefix="eval/",
                )
                logging.info(val_metrics, extra=dict(metrics=True))
                logging.info(val_metrics)

            logging.info(train_metrics, extra=dict(metrics=True))
            logging.info(train_metrics)

        # if accelerator.is_main_process and (step + 1) % save_steps == 0:
        #     checkpoint_path = (
        #         Path(log_dir)
        #         / f"checkpoint-{step + 1}"
        #         / ClassificationTuner.WEIGHTS_NAME
        #     )
        #     checkpoint_path.parent.mkdir()

        #     torch.save(accelerator.unwrap_model(model).state_dict(), checkpoint_path)

    model.eval()

    with accelerator.main_process_first():
        data_splits = get_dataset(
            f"offline:{model_name}",
            root=data_dir,
            seed=seed,
            use_cache=use_dataset_cache,
            prompt_style=prompt_style,
        )
    data_splits = [
        (s, ds)
        for s, ds in zip(["train", "validation", "test"], data_splits)
        if ds is not None
    ]

    offline_data = {"train": [], "validation": []}
    for split, data in data_splits:
        for d in data:
            offline_data[split].append((d["embedding"], -1))

    offline_data = {k: CategoryDataset(v) for k, v in offline_data.items()}

    classified_data = {"train": [], "validation": []}

    for split, data in offline_data.items():
        loader = get_loader(
            data,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            accelerator=accelerator,
            shuffle=True,
        )

        for embeddings, labels in tqdm(loader):
            classified_data[split].append(model(embeddings).argmax(dim=-1).cpu())

    classified_data = {k: torch.cat(v, dim=0) for k, v in classified_data.items()}

    # with open("offline.bin", "wb") as f:
    #     torch.save(classified_data, f)

    logging.info(slabels)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
