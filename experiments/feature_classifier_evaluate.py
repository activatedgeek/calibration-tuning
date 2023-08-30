from tqdm.auto import tqdm
import logging
import wandb
import torch
import pandas as pd
from accelerate import Accelerator
from peft import PeftModel


from llm.logging import entrypoint
from llm.models import get_model, get_special_tokens
from llm.utils.trainer import get_last_checkpoint_path
from llm.datasets import get_dataset, get_loader
from llm.datasets.llm_utils import (
    DataCollatorForSupervisedDataset,
)
from llm.utils.evaluation import extract_eos_pos
from llm.utils.third_party.calibration import calibration

def prepare_model(
    causal_lm,
    accelerator,
    model_name,
    tokenizer,
    special_token_count,
    model_dir,
    peft_dir,
    fp8,
):
    model = get_model(
        model_name,
        device_map={"": accelerator.local_process_index},
        load_in_8bit=fp8,
        model_dir=model_dir,
        causal_lm=causal_lm,
    )

    model.resize_token_embeddings(len(tokenizer))
    if special_token_count:
        input_embeddings = model.get_input_embeddings().weight.data

        input_embeddings[-special_token_count:] = input_embeddings[
            :-special_token_count
        ].mean(dim=0, keepdim=True)

        if causal_lm:
            output_embeddings = model.get_output_embeddings().weight.data

            output_embeddings[-special_token_count:] = output_embeddings[
                :-special_token_count
            ].mean(dim=0, keepdim=True)

    # model = prepare_model_for_kbit_training(model)

    # if peft_dir is not None:
    #     peft_dir = get_last_checkpoint_path(peft_dir)

    #     model = PeftModel.from_pretrained(model, peft_dir)

    #     logging.info(f"Loaded PEFT checkpoint from '{peft_dir}'")

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

        eos_idx = extract_eos_pos(tokenizer, labels)
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
):
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
        model_dir=model_dir,
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
        peft_dir=peft_dir,
        fp8=fp8,
    )

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

    all_metrics = map(
        lambda m: {**m, **config, "dataset": dataset},
        list(filter(lambda m: m is not None, [val_metrics, test_metrics])),
    )
    logging.info(
        {"metrics": wandb.Table(dataframe=pd.DataFrame(all_metrics))},
        extra=dict(metrics=True),
    )


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint(main))
