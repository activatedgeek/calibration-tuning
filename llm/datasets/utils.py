import torch
from torch.utils.data import DataLoader, random_split


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


def get_num_workers(num_workers=4):
    num_gpus_per_host = torch.cuda.device_count()
    if num_gpus_per_host == 0:
        return num_workers
    return (num_workers + num_gpus_per_host - 1) // num_gpus_per_host


def get_loader(dataset, batch_size=128, num_workers=4, accelerator=None, **kwargs):
    num_workers = get_num_workers(num_workers=num_workers)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, **kwargs
    )
    if accelerator is not None:
        loader = accelerator.prepare(loader)

    return loader
