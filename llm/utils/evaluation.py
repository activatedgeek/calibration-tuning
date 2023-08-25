from tqdm.auto import tqdm
import torch

from .third_party.calibration import calibration
from ..datasets import get_dataset, get_loader
from ..datasets.llm_utils import (
    tokenize_for_causal_lm,
    DataCollatorForSupervisedDataset,
)


def extract_eos_pos(tokenizer, labels):
    """
    Extracts the position of the last EOS token from each row.
    """
    eos_idx = labels.eq(tokenizer.eos_token_id).nonzero()[
        labels.eq(tokenizer.eos_token_id).sum(dim=-1).cumsum(dim=0) - 1
    ][:, -1]

    return eos_idx


@torch.inference_mode()
def evaluate_via_eos(accelerator, model, tokenizer, loader):
    """
    Assumes all answers are 1 token and end immediately with EOS token.
    """
    device = accelerator.device

    uq_ans_token_vec = torch.tensor(
        [tokenizer("no").input_ids[-1], tokenizer("yes").input_ids[-1]]
    ).to(device)

    Y, P_hat = [], []
    UNC_Y, UNC_P_hat = [], []

    for inputs in tqdm(loader, leave=False):
        inputs = {k: v.to(device) for k, v in inputs.items()}

        labels = inputs.pop("labels")[..., 1:]

        logits = model(**inputs).logits[..., :-1, :]

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
        unc_prompts = tokenizer.batch_decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        unc_samples = [
            {
                "source": u + "\n" + "Is the proposed answer correct? ",
                "target": f"{'yes' if r else 'no'}{tokenizer.eos_token}",
            }
            for u, r in zip(unc_prompts, targets)
        ]
        tokenized_unc_samples = [
            tokenize_for_causal_lm(tokenizer, sample) for sample in unc_samples
        ]
        unc_inputs = {
            k: v.to(device) for k, v in loader.collate_fn(tokenized_unc_samples).items()
        }

        unc_labels = unc_inputs.pop("labels")
        unc_eos_idx = extract_eos_pos(tokenizer, unc_labels)

        unc_labels = unc_labels[torch.arange(unc_labels.size(0)), unc_eos_idx - 1]
        unc_y = (unc_labels.unsqueeze(-1) == uq_ans_token_vec).long().argmax(dim=-1)

        unc_logits = model(**unc_inputs).logits[
            torch.arange(unc_labels.size(0)), unc_eos_idx - 1, :
        ]
        unc_p_hat = unc_logits[..., uq_ans_token_vec]

        (__unc_y, __uq_p_hat) = accelerator.gather_for_metrics((unc_y, unc_p_hat))
        UNC_Y.append(__unc_y), UNC_P_hat.append(__uq_p_hat)

        ######### UQ Metrics #######

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
    model,
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
        val_metrics = evaluate_via_eos(
            accelerator,
            model,
            tokenizer,
            get_loader(
                val_data,
                batch_size=batch_size,
                collate_fn=DataCollatorForSupervisedDataset(tokenizer),
                accelerator=accelerator,
            ),
        )
        val_metrics["seed"] = seed
        val_metrics["eval_kshot"] = eval_kshot
        val_metrics["split"] = "validation"
        val_metrics["dataset"] = dataset

    test_metrics = None
    if test_data is not None:
        test_metrics = evaluate_via_eos(
            accelerator,
            model,
            tokenizer,
            get_loader(
                test_data,
                batch_size=batch_size,
                collate_fn=DataCollatorForSupervisedDataset(tokenizer),
                accelerator=accelerator,
            ),
        )
        test_metrics["seed"] = seed
        test_metrics["eval_kshot"] = eval_kshot
        test_metrics["split"] = "test"
        test_metrics["dataset"] = dataset

    return val_metrics, test_metrics
