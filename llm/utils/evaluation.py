from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

from .third_party.calibration import calibration


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

    Y, P_hat = [], []

    for inputs in tqdm(loader, leave=False):
        inputs = {k: v.to(device) for k, v in inputs.items()}

        labels = inputs.pop("labels")[..., 1:]

        logits = model(**inputs).logits[..., :-1, :]

        eos_idx = extract_eos_pos(tokenizer, labels)
        y = labels[torch.arange(labels.size(0)), eos_idx - 1]
        p_hat = logits[torch.arange(logits.size(0)), eos_idx - 1]

        (__y, __p_hat) = accelerator.gather_for_metrics((y, p_hat))
        Y.append(__y), P_hat.append(__p_hat)

    Y, P_hat = torch.cat(Y, dim=0), torch.cat(P_hat, dim=0).softmax(dim=-1)

    acc = (Y == P_hat.argmax(dim=-1)).float().mean()
    ece, _ = calibration(
        F.one_hot(Y, num_classes=P_hat.size(-1)).cpu().numpy(), P_hat.cpu().numpy()
    )

    return {
        "acc": acc.item(),
        "N": Y.size(0),
        "ece": ece,
    }
