import torch
from tqdm.auto import tqdm


def extract_aligned_eos_pos(tokenizer, shifted_labels):
    """
    Assumes labels are for CausalLM, shifted by 1 to the right.
    and extracts the position of the last EOS token.
    """
    eos_idx = shifted_labels.eq(tokenizer.eos_token_id).nonzero()[
        shifted_labels.eq(tokenizer.eos_token_id).sum(dim=-1).cumsum(dim=0) - 1
    ][:, -1]

    return eos_idx


@torch.no_grad()
def evaluate(accelerator, model, tokenizer, loader, do_sample=False, max_new_tokens=1):
    device = accelerator.device

    N = torch.tensor(0).long().to(device)
    N_acc = torch.tensor(0).long().to(device)

    for inputs in tqdm(loader, leave=False):
        shifted_labels = inputs.get("labels")[:, 1:]
        eos_idx = extract_aligned_eos_pos(tokenizer, shifted_labels)

        targets = shifted_labels[torch.arange(shifted_labels.size(0)), eos_idx - 1]

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

        _N = torch.tensor(len(targets)).long().to(accelerator.device)
        _N_acc = (
            torch.tensor(sum([r == l for r, l in zip(responses, targets)]))
            .long()
            .to(accelerator.device)
        )

        N += accelerator.gather(_N).sum()
        N_acc += accelerator.gather(_N_acc).sum()

    metrics = {"exact_match_acc": N_acc.item() / N.item(), "N": N.item()}

    return metrics
