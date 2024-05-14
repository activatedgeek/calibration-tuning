import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset

from ..registry import register_dataset


class LogitsDataset(Dataset):
    def __init__(self, logits, labels):
        self.logits = logits
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.logits[index], self.labels[index]


def get_offline_logits(root=None, **_):
    data_splits = dict()
    for split in ["train", "validation", "test"]:
        data_path = Path(root) / split

        if not data_path.is_dir():
            continue

        # data = [
        #     torch.load(path, map_location="cpu")["fuzzy_gpt-3.5-turbo-1106"]
        #     for path in data_path.glob("*.pt")
        # ]
        # data = {k: torch.cat([v[k] for v in data], dim=0) for k in data[0].keys()}
        # with open(data_path / "metrics.bin", "wb") as f:
        #     torch.save(data, f)

        try:
            data_path = next(data_path.glob("*.bin"))
        except StopIteration:
            logging.exception(f".bin file not found at {data_path}")
            raise

        data = torch.load(data_path, map_location="cpu")

        # if "fuzzy_gpt-3.5-turbo-1106" in data:
        #     data = data["fuzzy_gpt-3.5-turbo-1106"]
        #     with open(data_path.parent / "data.bin", "wb") as f:
        #         torch.save(data, f)
        #     logging.info(f"Saved {data_path}")

        logits = data.pop("q_logits")
        labels = data.pop("q_labels").long()

        data_splits[split] = LogitsDataset(logits, labels)

    train_data = data_splits.pop("train", None)
    val_data = data_splits.pop("validation", None)
    test_data = data_splits.pop("test", None)

    return train_data, val_data, test_data


@register_dataset(attrs=dict(unlisted=True))
def offline_logits(*args, root=None, dataset_str=None, **kwargs):
    try:
        _, kind = dataset_str.split(":")
    except ValueError:
        logging.exception(f"Dataset format should be offline_logits:<kind>.")
        raise

    root = Path(root) / "offline_logits" / kind

    return get_offline_logits(*args, root=root**kwargs)
