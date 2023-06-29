import logging
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import DataCollatorWithPadding

from llm.logging import set_logging
from llm.datasets import get_dataset, get_loader
from llm.models import create_model


def main(
    accelerator,
    seed=None,
    log_dir=None,
    data_dir=None,
    model_dir=None,
    dataset=None,
    batch_size=1,
    model_name=None,
    max_new_tokens=20,
    top_p=0.95,
):
    tokenizer = create_model(
        model_name=f"{model_name}_tokenizer", model_kwargs=dict(cache_dir=model_dir)
    )
    _, val_data, test_data = get_dataset(
        dataset,
        root=data_dir,
        tokenizer=tokenizer,
        seed=seed,
    )

    model = create_model(
        model_name=model_name,
        model_kwargs=dict(
            device_map={"": accelerator.device},
            cache_dir=model_dir,
        ),
    ).eval()
    model = accelerator.prepare(model)

    val_loader = get_loader(
        val_data,
        batch_size=batch_size,
        accelerator=accelerator,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    for inputs in tqdm(val_loader):
        outputs = accelerator.unwrap_model(model).generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p
        )
        responses = tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logging.debug(f"{len(responses)} responses")


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
