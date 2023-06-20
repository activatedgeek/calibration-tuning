import os
import logging
from tqdm.auto import tqdm
import torch.optim as optim
from accelerate import Accelerator
from transformers import LlamaTokenizer, get_scheduler

from llm.logging import set_logging
from llm.datasets import get_dataset, get_dataset_attrs, get_loader
from llm.models import create_model


def train_epoch(accelerator, model, loader, optimizer, epoch=None):
    device = accelerator.device

    model.train()

    for i, B in tqdm(enumerate(loader), leave=False):
        optimizer.zero_grad()

        loss = model(**{k: v.to(device) for k, v in B.items()})

        accelerator.backward(loss)

        optimizer.step()

        if accelerator.is_main_process and i % 100 == 0:
            metrics = {"epoch": epoch, "mini_loss": loss.detach().item()}
            logging.info(metrics, extra=dict(metrics=True, prefix="train"))
            logging.debug(metrics)


def main(
    accelerator,
    seed=None,
    log_dir=None,
    data_dir=None,
    dataset=None,
    batch_size=8,
    model_name=None,
    lr=5e-2,
    weight_decay=2e-5,
    epochs=0,
):
    ## @FIXME: parametrize.
    tokenizer = LlamaTokenizer.from_pretrained(
        "openlm-research/open_llama_13b", cache_dir=os.environ.get("MODELDIR")
    )
    tokenizer.pad_token = tokenizer.eos_token

    train_data, _, test_data = get_dataset(
        dataset,
        root=data_dir,
        tokenize_fn=lambda x: tokenizer(
            x["text"], padding="max_length", truncation=True
        ),
        seed=seed,
    )
    train_loader = get_loader(
        train_data, batch_size=batch_size, shuffle=True, accelerator=accelerator
    )
    test_loader = get_loader(test_data, batch_size=batch_size, accelerator=accelerator)

    model = create_model(
        num_classes=get_dataset_attrs(dataset).get("num_classes"),
        model_name=model_name,
        model_kwargs=dict(device_map={"": accelerator.local_process_index}),
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optim_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epochs * len(train_loader),
    )

    # model, optimizer, optim_scheduler = accelerator.prepare(model, optimizer, optim_scheduler)

    for e in tqdm(range(epochs)):
        train_epoch(accelerator, model, train_loader, optimizer, epoch=e)

        optim_scheduler.step()

    #### @FIXME: old HF docs copy.

    # metric = evaluate.load("accuracy")

    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis=-1)
    #     return metric.compute(predictions=predictions, references=labels)


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
