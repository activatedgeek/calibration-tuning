import logging
from pathlib import Path
import os
import wandb
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import GenerationConfig

from llm.datasets import (
    get_all_datasets_list,
    get_dataset,
    get_loader,
    prepare_uncertainty_query,
    LMText,
)
from llm.logging import entrypoint
from llm.models import get_model
from llm.utils.generate_utils import generate_output
from llm.eval.utils import _dataset_log_name


@entrypoint(with_accelerator=True)
def generate_outputs_main(
    accelerator=None,
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    prompt_style=None,
    kshot=0,
    use_dataset_cache=True,
    batch_size=1,
    model_name=None,
    max_new_tokens=30,
):
    config = {
        "seed": seed,
        "log_dir": log_dir,
        "dataset": dataset,
        "prompt_style": prompt_style,
        "kshot": kshot,
        "batch_size": batch_size,
        "model_name": model_name,
        "max_new_tokens": max_new_tokens,
    }
    if accelerator.is_main_process:
        wandb.config.update(config)

    ## @HOTFIX: for hanging processes in dataset map.
    # import multiprocess.context as ctx
    # ctx._force_start_method("spawn")
    with accelerator.main_process_first():
        data_splits = get_dataset(
            dataset,
            root=data_dir,
            seed=seed,
            num_workers=8,
            use_cache=use_dataset_cache,
            prompt_style=prompt_style,
            train_kshot=kshot,
            eval_kshot=kshot,
        )
        data_splits = [
            (s, ds)
            for s, ds in zip(["train", "validation", "test"], data_splits)
            if ds is not None
        ]

    tokenizer, model = get_model(model_name, device_map="auto")
    model.eval()

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )

    for split_name, data in data_splits:
        with accelerator.main_process_first():
            if accelerator.is_main_process:
                os.makedirs(f"{log_dir}/outputs/{split_name}")

        # import pandas as pd
        # pd.DataFrame([data[i] for i in tqdm(range(len(data)))]).to_csv(
        #     f"{log_dir}/outputs/{split_name}/rows.csv", index=False
        # )

        generate_output(
            accelerator,
            model,
            tokenizer,
            get_loader(
                data,
                batch_size=batch_size,
                pin_memory=True,
                accelerator=accelerator,
            ),
            generation_config=generation_config,
            log_dir=f"{log_dir}/outputs/{split_name}",
        )


def generate_query_label(
    accelerator,
    tokenizer,
    loader,
    query_format="roman_choice",
    strategy="substring",
):
    for inputs in tqdm(loader):
        inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
        targets = [inp.pop("target") for inp in inputs]
        outputs = [inp.pop("output") for inp in inputs]

        _, query_labels, _ = prepare_uncertainty_query(
            tokenizer,
            inputs,
            targets,
            outputs,
            format=query_format,
            strategy=strategy,
        )

        outputs = [
            {**inp, "target": tgt, "output": out, "query_label": int(ql.item())}
            for inp, tgt, out, ql in zip(inputs, targets, outputs, query_labels)
        ]

        yield from outputs


@entrypoint(with_accelerator=True)
def generate_labels_main(
    accelerator=None,
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    prompt_style=None,
    kshot=0,
    use_dataset_cache=True,
    batch_size=1,
    model_name=None,
    strategy=None,  ## substring / fuzzy_gpt-3.5-turbo-1106
):
    config = {
        "seed": seed,
        "log_dir": log_dir,
        "dataset": dataset,
        "prompt_style": prompt_style,
        "batch_size": batch_size,
        "model_name": model_name,
        "strategy": strategy,
    }
    if accelerator.is_main_process:
        wandb.config.update(config)

    tokenizer = get_model(f"{model_name}_tokenizer")

    with accelerator.main_process_first():
        data_splits = get_dataset(
            dataset,
            root=data_dir,
            seed=seed,
            num_workers=8,
            use_cache=use_dataset_cache,
            prompt_style=prompt_style,
            train_kshot=kshot,
            eval_kshot=kshot,
        )
        data_splits = [
            (s, ds)
            for s, ds in zip(["train", "validation", "test"], data_splits)
            if ds is not None
        ]

    for split_name, data in data_splits:
        loader = get_loader(
            data,
            batch_size=batch_size,
            pin_memory=True,
            accelerator=accelerator,
        )

        label_generator = generate_query_label(
            accelerator,
            tokenizer,
            loader,
            strategy=strategy,
        )

        csv_path = f"{log_dir}/labels/{split_name}"
        with accelerator.main_process_first():
            if accelerator.is_main_process:
                os.makedirs(csv_path)

        pd.DataFrame(label_generator).to_csv(
            f"{csv_path}/{accelerator.process_index}.csv", index=False
        )


@entrypoint(with_accelerator=True)
def generate_embeddings_main(
    accelerator=None,
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    use_dataset_cache=True,
    batch_size=1,
    model_name=None,
):
    config = {
        "seed": seed,
        "log_dir": log_dir,
        "dataset": dataset,
        "batch_size": batch_size,
        "model_name": model_name,
    }
    if accelerator.is_main_process:
        wandb.config.update(config)

    embedding_model = get_model(model_name)

    if dataset.startswith("eval"):
        all_datasets = get_all_datasets_list(dataset)
    else:
        assert dataset is not None, "Missing dataset."
        all_datasets = [dataset]

    for dataset in tqdm(all_datasets):
        with accelerator.main_process_first():
            data_splits = get_dataset(
                dataset,
                root=data_dir,
                seed=seed,
                num_workers=8,
                use_cache=use_dataset_cache,
            )
            data_splits = [
                (s, ds)
                for s, ds in zip(["train", "validation", "test"], data_splits)
                if ds is not None
            ]

        for split_name, data in tqdm(data_splits, leave=False):
            loader = get_loader(
                data,
                batch_size=batch_size,
                pin_memory=True,
                accelerator=accelerator,
            )

            all_embeddings = []

            for inputs in tqdm(loader, leave=False):
                inputs = [
                    dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())
                ]
                outputs = [inp.pop("output") for inp in inputs]
                targets = [inp.pop("target") for inp in inputs]
                query_labels = [inp.pop("query_label") for inp in inputs]

                texts = [
                    str(LMText.from_({**i, "target": o}))
                    for i, o in zip(inputs, outputs)
                ]

                # embeddings = np.random.randn(len(inputs), 100)
                embeddings = embedding_model(texts)
                embeddings = torch.tensor(embeddings).float()

                all_embeddings.append(accelerator.gather_for_metrics(embeddings))

            all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

            save_dir = (
                Path(log_dir) / "embeddings" / _dataset_log_name(dataset) / split_name
            )
            save_dir.mkdir(parents=True)

            np.save(save_dir / "embedding.npy", all_embeddings)
            logging.info(f"Saved embeddings to '{save_dir}'.")


if __name__ == "__main__":
    import fire

    fire.Fire(
        dict(
            outputs=generate_outputs_main,
            labels=generate_labels_main,
            embeds=generate_embeddings_main,
        )
    )
