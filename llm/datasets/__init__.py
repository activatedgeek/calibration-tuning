from .registry import register_dataset, get_dataset, get_dataset_attrs, list_datasets
from .utils import IndexedDataset, LabelNoiseDataset, get_loader

__all__ = [
    "register_dataset",
    "get_dataset",
    "get_dataset_attrs",
    "list_datasets",
    "IndexedDataset",
    "LabelNoiseDataset",
    "get_loader",
]


def __setup():
    from importlib import import_module

    for n in [
        "trivia_qa",
        "truthful_qa",
        "yelp",
    ]:
        import_module(f".{n}", __name__)


__setup()
