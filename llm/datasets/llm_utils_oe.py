from openai import ChatCompletion
from openai.error import RateLimitError, APIError, Timeout, ServiceUnavailableError
import torch

from .llm_utils import (
    get_token_vec,
    IGNORE_LABEL,
    tokenize_for_causal_lm
)


def openai_query(system_prompt, prompt, openai_model_name='gpt-4-0314'):
    sampled_response = None
    while sampled_response is None:
        try:                
            response = ChatCompletion.create(
                model=openai_model_name,
                messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                )
            sampled_response = response["choices"][0]["message"]["content"]
        except (RateLimitError, APIError, Timeout, ServiceUnavailableError) as e:
            logging.info(f'Possible OpenAI rate limit: {e}')
            time.sleep(1)
    return sampled_response


SYSTEM_PROMPT_ORACLE_EQUIVALENCY    = "You are an automated grading assistant helping a teacher grade student answers."
PROMPT_ORACLE_EQUIVALENCY           = "The correct answer for this problem is: <ground-truth>\n. " + \
        "A student submitted the answer: <prediction>\n. " + \
        "The student's answer must be correct and specific but not overcomplete " + \
        "(for example, if they provide two different answers, they did not get the question right). " + \
        "However, small differences in formatting should not be penalized (for example, 'New York City' is equivalent to 'NYC'). " + \
        "Did the student provide an equivalent answer to the ground truth? Please answer yes or no without any explanation: "


def evaluate_equivalency_with_oracle(ground_truth, prediction, oracle_fn, oracle_kwargs):
    prompt = PROMPT_ORACLE_EQUIVALENCY.replace('<ground-truth>', ground_truth).replace('<prediction>', prediction)
    sampled_response = oracle_fn(
        system_prompt=SYSTEM_PROMPT_ORACLE_EQUIVALENCY,
        prompt=prompt,
        **oracle_kwargs
    )
    return 'yes' in sampled_response.strip().lower()


def extract_qa_oe(tokenizer, inputs, outputs=None):
    """
    TBD: for @arka to adapt
    Assumes all answers are open ended and end with EOS token.
    """
    labels = inputs.get("labels")[..., 1:]

    eos_idx = labels.eq(tokenizer.eos_token_id).nonzero()[
        labels.eq(tokenizer.eos_token_id).sum(dim=-1).cumsum(dim=0) - 1
    ][:, -1]

    y = labels[torch.arange(labels.size(0)), :]
    

    if outputs is not None:
        logits = outputs.logits[..., :-1, :]
        logits = logits[torch.arange(logits.size(0)), :]

        return eos_idx, y, logits

    return eos_idx, y


def extract_oe_inputs(tokenizer, inputs):
    target_start_idx = (
        inputs.get("labels")
        .eq(-100)
        .nonzero()[
            inputs.get("labels").eq(IGNORE_LABEL).sum(dim=-1).cumsum(dim=-1) - 1
        ][:, -1]
        + 1
    )

    oe_inputs = [
        tokenizer(
            tokenizer.decode(inp[1:t].tolist()),
            padding="longest",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        for inp, t in zip(inputs.get("input_ids"), target_start_idx)
    ]

    oe_targets = torch.cat(
        [
            inp[t:].unsqueeze(0)
            for inp, t in zip(inputs.get("input_ids"), target_start_idx)
        ],
        dim=0,
    )

    return target_start_idx, oe_inputs, oe_targets


def prepare_oe_calibration_query(tokenizer, true, pred, format="roman_choice", comparison_strategy='substring'):

    # calculate accuracy
    if comparison_strategy == 'substring':
        comparison_fn = lambda t,p: t in p
    elif comparison_strategy == 'fuzzy_gpt4':
        comparison_fn = lambda t,p: evaluate_equivalency_with_oracle(
            t,
            p,
            oracle_fn=openai_query,
            oracle_kwargs={'openai_model_name': 'gpt-4-0314'}
        )
    else:
        raise ValueError(f'Invalid comparison strategy {comparison_strategy}') 
    acc = [comparison_fn(t,p) for t, p in zip(true, pred)]

    if format == "bool":
        ## NOTE: Probably don't use, often seems to be biased towards a yes.
        query_inputs = [
            tokenize_for_causal_lm(
                tokenizer,
                {
                    "context": f"{r}\n\nIs the proposed answer correct? ",
                    "target": ("yes" if a else "no") + tokenizer.eos_token,
                },
                prompt_style="choice",
            )
            for r, a in zip(pred, acc)
        ]
    elif format == "alpha_choice":
        query_inputs = [
            tokenize_for_causal_lm(
                tokenizer,
                {
                    "context": f"{r}\n\nIs the proposed answer correct?\n\nChoices:\n(a): no\n(b): yes",
                    "target_prompt": "\nAnswer: ",
                    "target": ("b" if a else "a") + tokenizer.eos_token,
                },
                prompt_style="choice",
            )
            for r, a in zip(pred, acc)
        ]
    elif format == "roman_choice":
        query_inputs = [
            tokenize_for_causal_lm(
                tokenizer,
                {
                    "context": f"{r}\n\nIs the proposed answer correct?\n\nChoices:\n(i): no\n(ii): yes",
                    "target_prompt": "\nAnswer: ",
                    "target": ("ii" if a else "i") + tokenizer.eos_token,
                },
                prompt_style="choice",
            )
            for r, a in zip(pred, acc)
        ]
    else:
        raise NotImplementedError

    return query_inputs, get_token_vec(tokenizer, format=format), torch.Tensor(
            [1 if a else 0 for a in acc]
        )
