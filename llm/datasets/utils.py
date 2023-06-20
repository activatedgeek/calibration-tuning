import os
import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.distributions import Categorical


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (*self.dataset[index], index)


class WrapperDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

    @property
    def targets(self):
        if hasattr(self.dataset, "targets"):
            return self.dataset.targets

        return torch.Tensor([y for _, y in self.dataset])

    @targets.setter
    def targets(self, __value):
        return setattr(self.dataset, "targets", __value)

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, __value):
        return setattr(self.dataset, "transform", __value)

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class LabelNoiseDataset(WrapperDataset):
    def __init__(self, dataset, n_labels=10, label_noise=0, seed=42):
        super().__init__(dataset)

        self.C = n_labels

        if label_noise > 0:
            orig_targets = self.targets
            rv = torch.rand(
                len(orig_targets), generator=torch.Generator().manual_seed(seed)
            )
            self.noisy_targets = torch.where(
                rv < label_noise,
                Categorical(probs=torch.ones(self.C) / self.C).sample(
                    torch.Size([len(orig_targets)])
                ),
                torch.Tensor(orig_targets).long(),
            )

    def __getitem__(self, i):
        X, y = super().__getitem__(i)
        y = self.noisy_targets[i]
        return X, y


def get_data_dir(data_dir=None):
    if data_dir is None:
        if os.environ.get("DATADIR") is not None:
            data_dir = os.environ.get("DATADIR")
            logging.debug(
                f'Using default data directory from environment "{data_dir}".'
            )
        else:
            home_data_dir = Path().home() / "datasets"
            data_dir = str(home_data_dir.resolve())
            logging.debug(f'Using default HOME data directory "{data_dir}".')

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    return data_dir


def train_test_split(dataset, test_size=0.2, seed=None):
    N = len(dataset)
    N_test = int(test_size * N)
    N -= N_test

    if seed is not None:
        train, test = random_split(
            dataset, [N, N_test], generator=torch.Generator().manual_seed(seed)
        )
    else:
        train, test = random_split(dataset, [N, N_test])

    return train, test


def get_loader(dataset, batch_size=128, num_workers=4, accelerator=None, **kwargs):
    num_gpus_per_host = torch.cuda.device_count()
    num_workers = (num_workers + num_gpus_per_host - 1) // num_gpus_per_host

    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, **kwargs
    )
    if accelerator is not None:
        loader = accelerator.prepare(loader)

    return loader
