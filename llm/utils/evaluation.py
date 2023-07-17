import torch
from tqdm.auto import tqdm


def extract_eos_pos(tokenizer, labels):
    """
    Assumes labels are for CausalLM, shifted by 1 to the right.
    and extracts the position of the last EOS token.
    """
    eos_idx = labels.eq(tokenizer.eos_token_id).nonzero()[
        labels.eq(tokenizer.eos_token_id).sum(dim=-1).cumsum(dim=0) - 1
    ][:, -1]

    return eos_idx


@torch.no_grad()
def evaluate_via_eos(accelerator, model, tokenizer, loader):
    """
    Assumes all answers are 1 token and end immediately with EOS token.
    """
    device = accelerator.device

    acc = []

    for inputs in tqdm(loader, leave=False):
        inputs = {k: v.to(device) for k, v in inputs.items()}

        labels = inputs.pop("labels")[..., 1:]

        logits = model(**inputs).logits[..., :-1, :]
        outputs = logits.argmax(dim=-1)

        eos_idx = extract_eos_pos(tokenizer, labels)
        targets = labels[torch.arange(labels.size(0)), eos_idx - 1]
        responses = outputs[torch.arange(outputs.size(0)), eos_idx - 1]

        (__acc,) = accelerator.gather_for_metrics((targets == responses,))
        acc.append(__acc)
    
    acc = torch.cat(acc)

    metrics = {"match_acc": acc.float().mean().item(), "N": acc.size(0)}

    return metrics
