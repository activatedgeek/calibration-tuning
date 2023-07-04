import logging
from tqdm.auto import tqdm
import torch
from accelerate import Accelerator
import tensor_parallel as tp
from transformers import DataCollatorWithPadding

from llm.logging import set_logging
from llm.datasets import get_dataset, get_dataset_attrs, get_loader
from llm.models import create_model


@torch.no_grad()
def evaluate(
    accelerator, model, tokenizer, label2char, loader, do_sample=False, max_new_tokens=1
):
    device = accelerator.device

    N = torch.tensor(0).long().to(device)
    N_acc = torch.tensor(0).long().to(device)

    for inputs in tqdm(loader, leave=False):
        labels = [label2char(l) for l in inputs.pop("labels")]

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        responses = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        _N = torch.tensor(len(labels)).long().to(accelerator.device)
        _N_acc = (
            torch.tensor(sum([r == l for r, l in zip(responses, labels)]))
            .long()
            .to(accelerator.device)
        )

        N += accelerator.gather(_N).sum()
        N_acc += accelerator.gather(_N_acc).sum()

    metrics = {"exact_match_acc": N_acc.item() / N.item(), "N": N.item()}

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
    sample=False,
    max_new_tokens=1,
):
    tokenizer = create_model(
        model_name=f"{model_name}_tokenizer", model_kwargs=dict(cache_dir=model_dir)
    )

    label2char = get_dataset_attrs(dataset).get("label2char")
    _, val_data, test_data = get_dataset(
        dataset,
        instance=dataset_instance,
        root=data_dir,
        tokenizer=tokenizer,
        seed=seed,
    )

    model = create_model(
        model_name=model_name,
        model_kwargs=dict(
            cache_dir=model_dir,
        ),
    )
    ## NOTE: move inside create_model since all models may not work with this.
    model = tp.tensor_parallel(model, range(torch.cuda.device_count()))

    val_metrics = evaluate(
        accelerator,
        model,
        tokenizer,
        label2char,
        get_loader(
            val_data,
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(tokenizer),
        ),
        do_sample=sample,
        max_new_tokens=max_new_tokens,
    )
    logging.info(val_metrics, extra=dict(metrics=True, prefix="val"))

    test_metrics = evaluate(
        accelerator,
        model,
        tokenizer,
        label2char,
        get_loader(
            test_data,
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(tokenizer),
        ),
        do_sample=sample,
        max_new_tokens=max_new_tokens,
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

    # with FixedSeedAll(seed):
    main(accelerator, **kwargs, seed=seed, log_dir=log_dir)

    if accelerator.is_main_process:
        finish_logging()


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint)
