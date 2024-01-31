from concurrent.futures import ThreadPoolExecutor
import openai
from openai import ChatCompletion
from openai.error import RateLimitError, APIError, Timeout, ServiceUnavailableError
import torch
import logging
import time

from .llm_utils import (
    IGNORE_LABEL,
    get_token_vec,
    LMText,
)


def openai_query(system_prompt, prompt, openai_model_name="gpt-4-1106-preview"):
    # openai.api_requestor.TIMEOUT_SECS = 50
    sampled_response = None
    while sampled_response is None:
        try:
            response = ChatCompletion.create(
                model=openai_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            sampled_response = response["choices"][0]["message"]["content"]
        except (RateLimitError, APIError, Timeout, ServiceUnavailableError) as e:
            logging.info(f"Possible OpenAI rate limit: {e}")
            time.sleep(1)
    return sampled_response


SYSTEM_PROMPT_ORACLE_EQUIVALENCY = (
    "You are an automated grading assistant helping a teacher grade student answers."
)
PROMPT_ANSWER_KEY_EQUIVALENCY = (
    "The problem is: <question>\n\n The correct answer for this problem is: <ground-truth>\n "
    + "A student submitted the answer: <prediction>\n "
    + "The student's answer must be correct and specific but not overcomplete "
    + "(for example, if they provide two different answers, they did not get the question right). "
    + "However, small differences in formatting should not be penalized (for example, 'New York City' is equivalent to 'NYC'). "
    + "Did the student provide an equivalent answer to the ground truth? Please answer yes or no without any explanation: "
)

PROMPT_TWO_ANSWERS_EQUIVALENCY = (
    "The problem is: <question>\n\n"
    + "Student A submitted the answer: <prediction-a>\n "
    + "Student B submitted the answer: <prediction-b>\n"
    + "Your task is to evaluate if these two answers are equivalent, so the teacher can group matching answers together. "
    + "Small differences in formatting should not be a reason to mark the answers as different. "
    + "Did the two students provide equivalent answers? Please answer yes or no without any explanation: "
)


def evaluate_equivalency_with_oracle(
    ground_truth, prediction, question, oracle_fn, oracle_kwargs, mode="answer-key"
):
    if mode == "answer-key":
        prompt = (
            PROMPT_ANSWER_KEY_EQUIVALENCY.replace("<ground-truth>", ground_truth)
            .replace("<prediction>", prediction)
            .replace("<question>", question)
        )
    elif mode == "two-answers":
        prompt = (
            PROMPT_TWO_ANSWERS_EQUIVALENCY.replace("<prediction-a>", ground_truth)
            .replace("<prediction-b>", prediction)
            .replace("<question>", question)
        )
    else:
        raise NotImplementedError
    sampled_response = oracle_fn(
        system_prompt=SYSTEM_PROMPT_ORACLE_EQUIVALENCY, prompt=prompt, **oracle_kwargs
    )
    return "yes" in sampled_response.strip().lower()


def clustering_equivalency_with_oracle(a, b, question, oracle_fn, oracle_kwargs):
    prompt = (
        PROMPT_TWO_ANSWERS_EQUIVALENCY.replace("<prediction-a>", a)
        .replace("<prediction-b>", b)
        .replace("<question>", question)
    )
    sampled_response = oracle_fn(
        system_prompt=SYSTEM_PROMPT_ORACLE_EQUIVALENCY, prompt=prompt, **oracle_kwargs
    )
    return "yes" in sampled_response.strip().lower()


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


def grade_oe_preds(
    true,
    pred,
    questions,
    comparison_strategy="substring",
    mode="answer-key",
    max_threads=50
):
    # calculate accuracy
    if comparison_strategy == "substring":
        comparison_fn = lambda t, p, q: t in p
    elif "fuzzy" in comparison_strategy:
        comparison_fn = lambda t, p, q: evaluate_equivalency_with_oracle(
            t,
            p,
            q,
            oracle_fn=openai_query,
            oracle_kwargs={"openai_model_name": comparison_strategy.split("_")[1]},
            mode=mode,
        )
    else:
        raise ValueError(f"Invalid comparison strategy {comparison_strategy}")
    with ThreadPoolExecutor(min(max_threads, len(true))) as p:
        acc = list(
            p.map(comparison_fn, true, pred, questions) 
        )
    return acc


def newline_strip(
    input_string
):
    output_string = input_string
    output_string = output_string.replace("\n\n","\n")
    output_string = output_string.replace(":\n",":")
    output_string = output_string.strip("\n").split("\n")[0]
    return output_string


def equivalency_grading(
    inputs,
    targets,
    predictions,
    strategy="substring",
):
    contexts = [str(LMText.from_(inp)) for inp in inputs]

    query_labels = grade_oe_preds(
        targets, predictions, contexts, strategy, mode="two-answers"
    )

    return torch.Tensor(query_labels).long()


def prepare_oe_uncertainty_query(
    tokenizer,
    inputs,
    targets,
    predictions,
    query_labels=None,
    strategy="substring",
    format="roman_choice",
):
    query_token_vec = get_token_vec(tokenizer, format=format)

    contexts = [str(LMText.from_(inp)) for inp in inputs]

    query_labels = query_labels or grade_oe_preds(
        targets, predictions, contexts, strategy, mode="answer-key"
    )

    if format == "roman_choice":
        query_inputs = [
            {
                "context": f"{c + ' ' + p}\n\nIs the proposed answer correct?\nChoices:\n(i): no\n(ii): yes",
                "target_prompt": "\nAnswer:",
                # "target": ("ii" if l else "i"),
            }
            for c, p, l in zip(contexts, predictions, query_labels)
        ]
    else:
        raise NotImplementedError

    return (
        query_inputs,
        torch.Tensor(query_labels).long(),
        query_token_vec,
    )


def sanitize_generations(generations):
    def clean(g):
        g = g.replace("\n\n", "\n")
        g = g.replace(":\n", ":")
        g = g.strip("\n").split("\n")[0]
        return g

    return list(map(clean, generations))
