from tqdm.auto import tqdm
import logging
import wandb
import torch
import pandas as pd
from accelerate import Accelerator
from peft import PeftModel


from llm.logging import entrypoint, Timer
from llm.models import get_model, get_special_tokens
from llm.utils.trainer import get_last_checkpoint_path
from llm.datasets import (
    get_dataset, get_loader, get_all_train_datasets, get_all_eval_datasets
)
from llm.datasets.llm_utils import (
    tokenize_datasets,
    DataCollatorForSupervisedDataset,
)
# from llm.utils.evaluation import (
#     extract_eos_pos, evaluate_via_eos
# )
from llm.eval.third_party.calibration import calibration

def prepare_model(
    causal_lm,
    accelerator,
    model_name,
    tokenizer,
    special_token_count,
    model_dir,
    peft_dir,
    fp8,
    base_model=None,
):
    model = get_model(
        model_name,
        device_map={"": accelerator.local_process_index},
        load_in_8bit=fp8,
        model_dir=model_dir,
        causal_lm=causal_lm,
        tokenizer=tokenizer,
        base_model=base_model,
    )

    if peft_dir is not None:
        peft_dir = get_last_checkpoint_path(peft_dir)

        model = PeftModel.from_pretrained(model, peft_dir)

        logging.info(f"Loaded PEFT checkpoint from '{peft_dir}'")

    return model

@torch.inference_mode()
def evaluate_classifier(
    accelerator,
    base_model,
    classifier_model,
    tokenizer,
    loader,
):
    device = accelerator.device

    Y, P_hat = [], []
    UNC_Y, UNC_P_hat = [], []

    for inputs in tqdm(loader, leave=False):
        inputs = {k: v.to(device) for k, v in inputs.items()}

        labels = inputs.pop("labels")[..., 1:]
        attn_mask = inputs.get("attention_mask")

        logits = base_model(**inputs).logits[..., :-1, :]

        eos_idx = labels.eq(tokenizer.eos_token_id).nonzero()[
            labels.eq(tokenizer.eos_token_id).sum(dim=-1).cumsum(dim=0) - 1
        ][:, -1]

        y = labels[torch.arange(labels.size(0)), eos_idx - 1]
        p_hat = logits[torch.arange(logits.size(0)), eos_idx - 1]

        (__y, __p_hat) = accelerator.gather_for_metrics((y, p_hat))
        Y.append(__y), P_hat.append(__p_hat)

        ######### UQ Metrics #######

        output_ids = logits.argmax(dim=-1)

        targets = (
            labels[torch.arange(labels.size(0)), eos_idx - 1]
            == output_ids[torch.arange(output_ids.size(0)), eos_idx - 1]
        )

        response_ids = inputs.get("input_ids").clone()
        response_ids[torch.arange(response_ids.size(0)), eos_idx] = output_ids[
            torch.arange(output_ids.size(0)), eos_idx - 1
        ]

        unc_y = targets
        unc_p_hat = classifier_model(
            input_ids=response_ids,
            attention_mask=attn_mask,
            labels=torch.tensor(targets, device=targets.device).long(),
        ).logits

        (__unc_y, __uq_p_hat) = accelerator.gather_for_metrics((unc_y, unc_p_hat))
        UNC_Y.append(__unc_y), UNC_P_hat.append(__uq_p_hat)

    Y, P_hat = torch.cat(Y, dim=0), torch.cat(P_hat, dim=0).softmax(dim=-1)

    Y_hat = P_hat.argmax(dim=-1)
    acc = (Y == Y_hat).float().mean()
    ece, _ = calibration(Y, Y_hat, P_hat[torch.arange(Y_hat.size(0)), Y_hat])

    UNC_Y, UNC_P_hat = torch.cat(UNC_Y, dim=0), torch.cat(UNC_P_hat, dim=0).softmax(
        dim=-1
    )

    UNC_Y_hat = UNC_P_hat.argmax(dim=-1)
    UNC_acc = (UNC_Y == UNC_Y_hat).float().mean()
    UNC_ece, _ = calibration(
        UNC_Y, UNC_Y_hat, UNC_P_hat[torch.arange(UNC_Y_hat.size(0)), UNC_Y_hat]
    )

    ## Using confidence scores from "yes" (idx 1) always.
    qa_UNC_ece, _ = calibration(Y, Y_hat, UNC_P_hat[:, 1])

    return {
        "N": Y.size(0),
        "acc": acc.item(),
        "ece": ece,
        "unc_acc": UNC_acc.item(),
        "unc_ece": UNC_ece,
        "qa_unc_ece": qa_UNC_ece,
    }


def evaluate_dataset(
    accelerator,
    base_model,
    classifier_model,
    tokenizer,
    dataset,
    seed=137,
    batch_size=1,
    data_dir=None,
    eval_kshot=None,
    use_cache=True,
):
    with accelerator.main_process_first():
        _extra_args = dict()
        ## NOTE: Conditional to avoid overriding default kshot specification in dataset definition.
        if eval_kshot is not None:
            _extra_args["eval_kshot"] = eval_kshot
        _, val_data, test_data = get_dataset(
            dataset,
            root=data_dir,
            tokenizer=tokenizer,
            seed=seed,
            use_cache=use_cache,
            **_extra_args,
        )
        val_data, test_data = tokenize_datasets(tokenizer, val_data, test_data)

    val_metrics = None
    if val_data is not None:
        val_metrics = evaluate_classifier(
            accelerator,
            base_model,
            classifier_model,
            tokenizer,
            get_loader(
                val_data,
                batch_size=batch_size,
                collate_fn=DataCollatorForSupervisedDataset(tokenizer),
                accelerator=accelerator,
            ),
        )
        val_metrics["split"] = "validation"

        logging.debug(val_metrics)

    test_metrics = None
    if test_data is not None:
        test_metrics = evaluate_classifier(
            accelerator,
            base_model,
            classifier_model,
            tokenizer,
            get_loader(
                test_data,
                batch_size=batch_size,
                collate_fn=DataCollatorForSupervisedDataset(tokenizer),
                accelerator=accelerator,
            ),
        )
        test_metrics["split"] = "test"

        logging.debug(test_metrics)

    return val_metrics, test_metrics


def main(
    seed=137,
    log_dir=None,
    eval_kshot=None,
    dataset=None,
    data_dir=None,
    batch_size=1,
    model_name=None,
    fp8=True,
    model_dir=None,
    peft_dir=None,
    use_dataset_cache=True,
    **kwargs,
):
    # model_name = 'llama2_7b'
    # dataset = 'hellaswag'
    # peft_dir = '/data/home/ngruver/llm-calibration/checkpoint-17000'
    # model_dir = '/fsx-open-catalyst/ngruver/calibration_exps/checkpoint-6600'
    # log_dir = './eval_classifier'
    # batch_size = 1

    accelerator = Accelerator()

    config = {
        "seed": seed,
        "dataset": dataset,
        "model_name": model_name,
        "fp8": fp8,
        "model_dir": model_dir,
        "peft_dir": peft_dir,
        "eval_kshot": eval_kshot,
    }
    if accelerator.is_main_process:
        wandb.config.update(config)

    tokenizer = get_model(
        f"{model_name}_tokenizer",
        model_dir=model_dir,
    )
    special_token_count = tokenizer.add_special_tokens(get_special_tokens(tokenizer))

    base_model = prepare_model(
        causal_lm=True,
        accelerator=accelerator,
        model_name=model_name,
        tokenizer=tokenizer,
        special_token_count=special_token_count,
        model_dir=None,
        peft_dir=peft_dir,
        fp8=fp8,
    )

    classifier_model = prepare_model(
        causal_lm=False,
        accelerator=accelerator,
        model_name=model_name,
        tokenizer=tokenizer,
        special_token_count=special_token_count,
        model_dir=model_dir,
        peft_dir=None,
        fp8=fp8,
        base_model=base_model,
    )

    if dataset == "all":
        all_datasets = get_all_train_datasets() + get_all_eval_datasets()
    elif dataset == "eval":
        all_datasets = get_all_eval_datasets()
    else:
        assert dataset is not None, "Missing dataset."
        all_datasets = [dataset]

    all_metrics = []
    for dataset in tqdm(all_datasets):
        with Timer() as t:
            val_metrics, test_metrics = evaluate_dataset(
                accelerator,
                base_model,
                classifier_model,
                tokenizer,
                dataset,
                seed=seed,
                batch_size=batch_size,
                data_dir=data_dir,
                eval_kshot=eval_kshot,
                use_cache=use_dataset_cache,
            )

        dataset_metrics = list(
            map(
                lambda m: {**m, **config, "dataset": dataset, "ts": t.elapsed},
                list(filter(lambda m: m is not None, [val_metrics, test_metrics])),
            )
        )
        all_metrics += dataset_metrics
        logging.info(
            {"metrics": wandb.Table(dataframe=pd.DataFrame(all_metrics))},
            extra=dict(metrics=True),
        )

if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint(main))