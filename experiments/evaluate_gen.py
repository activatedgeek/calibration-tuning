import logging
from tqdm.auto import tqdm
import torch
from accelerate import Accelerator
from transformers import DataCollatorWithPadding

from llm.logging import set_logging
from llm.datasets import get_dataset, get_loader
from llm.datasets.llm_utils import DataCollatorForSupervisedDataset
from llm.models import create_model, get_special_tokens
from llm.utils.evaluation import extract_aligned_eos_pos


@torch.no_grad()
def evaluate(accelerator, model, tokenizer, loader):
    device = accelerator.device

    N = torch.tensor(0).long().to(device)
    N_acc = torch.tensor(0).long().to(device)

    for inputs in tqdm(loader, leave=False):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        shifted_labels = inputs.pop("labels")[..., 1:]

        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
        )[..., :-1]

        eos_idx = extract_aligned_eos_pos(tokenizer, shifted_labels)
        targets = shifted_labels[torch.arange(shifted_labels.size(0)), eos_idx - 1]
        responses = outputs[torch.arange(outputs.size(0)), eos_idx - 1]

        # if accelerator.is_main_process:
        #     print('REF *******************************************')
        #     print(inputs["input_ids"][..., -5:])
        #     print(tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)[0][-5:])
        #     print(shifted_labels[..., -5:])
        #     print('START *******************************************')
        #     print(outputs[..., -5:])
        #     print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][-5:])
        #     print('END *******************************************')

        #     assert False

        _N = torch.tensor(targets.size(0)).to(device)
        _N_acc = (targets == responses).sum()

        N += accelerator.gather(_N).sum()
        N_acc += accelerator.gather(_N_acc).sum()

    metrics = {"match_acc": N_acc.item() / N.item(), "N": N.item()}

    return metrics


def main(
    accelerator,
    seed=None,
    log_dir=None,
    data_dir=None,
    model_dir=None,
    dataset=None,
    dataset_instance=None,
    batch_size=1,
    model_name=None,
    fp8=True,
):
    tokenizer = create_model(
        model_name=f"{model_name}_tokenizer", model_kwargs=dict(cache_dir=model_dir)
    )
    special_token_count = tokenizer.add_special_tokens(get_special_tokens(tokenizer))

    train_data, val_data, test_data = get_dataset(
        dataset,
        instance=dataset_instance,
        root=data_dir,
        tokenizer=tokenizer,
        seed=seed,
    )

    model = create_model(
        model_name=model_name,
        model_kwargs=dict(
            device_map={"": accelerator.device},
            load_in_8bit=fp8,
            cache_dir=model_dir,
        ),
    )

    # train_metrics = evaluate(
    #     accelerator,
    #     model,
    #     tokenizer,
    #     get_loader(
    #         train_data,
    #         batch_size=batch_size,
    #         collate_fn=DataCollatorWithPadding(tokenizer),
    #     ),
    # )
    # logging.info(train_metrics, extra=dict(metrics=True, prefix="val"))

    val_metrics = evaluate(
        accelerator,
        model,
        tokenizer,
        get_loader(
            val_data,
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(tokenizer),
        ),
    )
    logging.info(val_metrics, extra=dict(metrics=True, prefix="val"))

    test_metrics = evaluate(
        accelerator,
        model,
        tokenizer,
        get_loader(
            test_data,
            batch_size=batch_size,
            collate_fn=DataCollatorForSupervisedDataset(tokenizer),
        ),
    )
    logging.info(test_metrics, extra=dict(metrics=True, prefix="test"))


def entrypoint(seed=None, log_dir=None, **kwargs):
    accelerator = Accelerator()

    ## Only setup logging from one process.
    log_dir, finish_logging = (
        set_logging(log_dir=log_dir) if accelerator.is_main_process else [None, None]
    )
    if accelerator.is_main_process:
        logging.info(f"Working with {accelerator.num_processes} process(es).")

    main(accelerator, **kwargs, seed=seed, log_dir=log_dir)

    if accelerator.is_main_process:
        finish_logging()


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint)
