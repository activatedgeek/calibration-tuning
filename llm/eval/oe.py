import logging
import os
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from peft import PeftModel
import pandas as pd
import numpy as np
from transformers import GenerationConfig
from sklearn.metrics import roc_auc_score

from llm.datasets import LabeledStringDataCollator
from llm.datasets.llm_utils_oe import (
    prepare_oe_uncertainty_query,
    equivalency_grading,
    sanitize_generations,
)
from llm.eval.third_party.calibration import calibration

# from llm.utils.generate_utils import generate_output
from llm.random import FixedSeed


@torch.inference_mode()
def evaluate_oe(
    accelerator,
    model,
    tokenizer,
    loader,
    query_format="roman_choice",
    comparison_strategies=None,
    max_new_tokens=100,
    log_dir=None,
    **_,
):
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    collate_fn = LabeledStringDataCollator(tokenizer)

    cs_q_labels = {c: [] for c in comparison_strategies}
    cs_q_logits = {c: [] for c in comparison_strategies}

    all_data = {
        "rows": [],
        "evals": {c: {"q_labels": [], "q_logits": []} for c in comparison_strategies},
    }

    for inputs in tqdm(loader):
        inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
        targets = [inp.pop("target") for inp in inputs]

        generation_inputs = {
            k: v.to(accelerator.device) for k, v in collate_fn(inputs).items()
        }

        if isinstance(model, PeftModel):
            model.set_adapter("default")

        generation_outputs = model.generate(
            **generation_inputs, generation_config=generation_config
        )

        generations = tokenizer.batch_decode(
            generation_outputs[:, generation_inputs.get("input_ids").size(-1) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        generations = sanitize_generations(generations)

        all_data["rows"].extend(
            [
                {**inp, "target": tgt, "output": out}
                for inp, tgt, out in zip(inputs, targets, generations)
            ]
        )

        if isinstance(model, PeftModel) and "query" in model.peft_config:
            model.set_adapter("query")

        for cs in comparison_strategies:
            q_inputs, q_labels, q_token_vec = prepare_oe_uncertainty_query(
                tokenizer,
                inputs,
                targets,
                generations,
                strategy=cs,
                format=query_format,
            )
            q_labels = q_labels.to(accelerator.device)
            # q_targets = [qi.pop("target") for qi in q_inputs]

            q_generation_inputs = {
                k: v.to(accelerator.device) for k, v in collate_fn(q_inputs).items()
            }

            q_generation_outputs = model(**q_generation_inputs)
            q_logits = q_generation_outputs.logits[..., -1, :]

            all_data["evals"][cs]["q_labels"].append(q_labels.detach())
            all_data["evals"][cs]["q_logits"].append(q_logits.detach())

            [
                l.append(v)
                for l, v in zip(
                    (cs_q_labels[cs], cs_q_logits[cs]),
                    accelerator.gather_for_metrics((q_labels, q_logits)),
                )
            ]

    metrics_dict = {}
    for cs in comparison_strategies:
        q_labels = torch.cat(cs_q_labels[cs], dim=0)
        q_p = torch.cat(cs_q_logits[cs], dim=0)[:, q_token_vec].softmax(dim=-1)

        acc = q_labels.float().mean(dim=0)
        q_pred = q_p.argmax(dim=-1)
        q_acc = (q_pred == q_labels).float().mean(dim=0)

        q_ece, _ = calibration(
            q_labels,
            q_pred,
            q_p[torch.arange(q_p.size(0)), q_pred],
        )

        try:
            q_auroc = roc_auc_score(
                q_labels.cpu(),
                q_p[torch.arange(q_p.size(0)), q_pred].cpu(),
                labels=np.array([0, 1]),
            )
        except ValueError:
            logging.warning(f"AUROC calculation failed.")
            q_auroc = float("nan")
        
        ece, _ = calibration(
            q_labels,
            q_pred,
            q_p[torch.arange(q_p.size(0)), 1],  ## corresponds to "yes"
        )

        metrics_dict.update(
            {
                "N": q_labels.size(0),
                f"{cs}_acc": acc.item(),
                f"{cs}_unc_acc": q_acc.item(),
                f"{cs}_unc_auroc": q_auroc,
                f"{cs}_unc_ece": q_ece,
                f"{cs}_ece": ece,
            }
        )

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

        all_data["evals"] = {
            k: {qk: torch.cat(qv, dim=0) for qk, qv in v.items()}
            for k, v in all_data["evals"].items()
        }

        pd.DataFrame(all_data["rows"]).to_csv(
            f"{log_dir}/rows_{accelerator.process_index}.csv", index=False
        )
        with open(f"{log_dir}/q_{accelerator.process_index}.pt", "wb") as f:
            torch.save(all_data["evals"], f)

        logging.debug(
            f"Data saved to '{log_dir}' from process {accelerator.process_index}."
        )

    return metrics_dict


@torch.inference_mode()
def evaluate_oe_uncertainty_sampling(
    accelerator,
    model,
    tokenizer,
    loader,
    prompt_style="oe",
    query_format="roman_choice",
    comparison_strategies=None,
    max_new_tokens=30,
    top_p=0.95,
    k=10,
    seed=1,
    log_dir=None,
    **_,
):
    assert prompt_style == "oe"
    assert (not comparison_strategies is None) and len(comparison_strategies) > 0

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generation_config_sampling = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        do_sample=True,
    )

    with FixedSeed(seed):
        collate_fn = LabeledStringDataCollator(tokenizer)

        cs_q_labels = {c: [] for c in comparison_strategies}

        cs_us_likelihood = {c: [] for c in comparison_strategies}
        cs_us_counting = {c: [] for c in comparison_strategies}

        all_data = {
            "rows": [],
            "evals": {
                c: {
                    "q_labels": [],
                    "summed_likelihood": [],
                    "sampling_count": [],
                    "sampling_equivalencies": [],
                }
                for c in comparison_strategies
            },
        }

        for inputs in tqdm(loader):
            inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
            targets = [inp.pop("target") for inp in inputs]

            generation_inputs = {
                k: v.to(accelerator.device) for k, v in collate_fn(inputs).items()
            }

            if isinstance(model, PeftModel):
                model.set_adapter("default")

            generation_outputs = model.generate(
                **generation_inputs, generation_config=generation_config
            )

            generations = tokenizer.batch_decode(
                generation_outputs[:, generation_inputs.get("input_ids").size(-1) :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            generations = sanitize_generations(generations)

            # page 15 is original algorithm: https://arxiv.org/pdf/2302.09664.pdf
            # our modification is based on finding the likelihood under the sampling procedure of the greedy decoded answer's equivalence class
            # form a cluster based on the equivalency with the greedy decode (but exclude the greedy decode)

            # two strategies to compute likelihood:
            # 1) find sum of length-normalized likelihoods of all entries in sample -- closest approximation of
            # the logsumpexp likelihood-summing done here:
            # https://github.com/lorenzkuhn/semantic_uncertainty/blob/27adbf0dc1bf056c771c205d89c2a79cbd82dc3a/code/compute_confidence_measure.py#L134
            # NOTE: we do not feel that adding likelihoods of samples produces a convergent estimate of likelihoods
            # 2) compute the size of the cluster associated with the greedy decode to get an estimate of its confidence - this is a convergent monte carlo estimate.

            # custom clustering procedure differs from paper; we are using modern LLMs for equivalency, not the NLI classifier used in the paper.
            # full prompting strategy is in llm/datasets/llm_utils_oe.py
            for cs in comparison_strategies:
                _, greedy_equivalency_labels, _ = prepare_oe_uncertainty_query(
                    tokenizer,
                    inputs,
                    targets,
                    generations,
                    strategy=cs,
                    format=query_format,
                )
                greedy_equivalency_labels = greedy_equivalency_labels.to(
                    accelerator.device
                )
                # q_targets = [qi.pop("target") for qi in q_inputs]

                sampling_likelihoods = []
                sampling_generations_list = []
                for _ in range(k):
                    sampling_generation_outputs = model.generate(
                        **generation_inputs,
                        generation_config=generation_config_sampling,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                    sampling_generations = tokenizer.batch_decode(
                        sampling_generation_outputs["sequences"][
                            :, generation_inputs.get("input_ids").size(-1) :
                        ],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    sampling_generations = sanitize_generations(sampling_generations)

                    # gets the max for each of the generated token among all log softmax probs in the vocab
                    sampling_log_probs = np.max(
                        F.log_softmax(
                            torch.stack(sampling_generation_outputs["scores"], dim=1),
                            dim=-1,
                        )
                        .detach()
                        .cpu()
                        .numpy(),
                        axis=-1,
                    )

                    # stop early at eos
                    eos_match_array = (
                        sampling_generation_outputs["sequences"][
                            :, generation_inputs.get("input_ids").size(-1) :
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                        == tokenizer.eos_token_id
                    )

                    has_eos = np.any(eos_match_array, axis=-1)

                    # we want to get the index after the last valid index for each row
                    stop_index = np.where(
                        has_eos,
                        # If there are multiple maximal values in a reduced row then the indices of the first maximal value are returned.
                        np.argmax(eos_match_array, axis=-1),
                        sampling_log_probs.shape[-1],
                    )

                    # negative inf are now set to 0 to allow summing, then summed, then divided by stop_index, then exponentiated
                    # assumption: each sample produces nonzero text
                    sampling_likelihood = np.exp(
                        np.sum(
                            np.where(
                                np.isinf(sampling_log_probs), 0, sampling_log_probs
                            ),
                            axis=-1,
                        )
                        / stop_index
                    )

                    sampling_likelihoods.append(sampling_likelihood)
                    sampling_generations_list.append(sampling_generations)

                sampling_equivalencies = np.reshape(
                    equivalency_grading(
                        inputs * k,
                        generations * k,
                        sum(sampling_generations_list, []),
                        strategy=cs,
                    )
                    .detach()
                    .cpu()
                    .numpy(),
                    (-1, k),
                )

                sampling_likelihoods = np.stack(sampling_likelihoods, axis=-1)
                normalized_sampling_likelihoods = sampling_likelihoods / np.sum(
                    sampling_likelihoods, axis=-1, keepdims=True
                )

                summed_likelihood = np.sum(
                    np.where(
                        sampling_equivalencies,
                        normalized_sampling_likelihoods,
                        0,
                    ),
                    axis=-1,
                )
                sampling_count = (
                    np.sum(sampling_equivalencies, axis=-1)
                    / sampling_equivalencies.shape[-1]
                )

                all_data["rows"].extend(
                    [
                        {
                            **inp,
                            "target": tgt,
                            "output": out,
                            "samples": sps,
                        }
                        for inp, tgt, out, sps in zip(
                            inputs, targets, generations, sampling_generations_list
                        )
                    ]
                )

                all_data["evals"][cs]["q_labels"].append(
                    greedy_equivalency_labels.detach().cpu().numpy()
                )
                all_data["evals"][cs]["summed_likelihood"].append(summed_likelihood)
                all_data["evals"][cs]["sampling_count"].append(sampling_count)
                all_data["evals"][cs]["sampling_equivalencies"].append(
                    sampling_equivalencies
                )

                [
                    l.append(v)
                    for l, v in zip(
                        (cs_q_labels[cs], cs_us_likelihood[cs], cs_us_counting[cs]),
                        accelerator.gather_for_metrics(
                            (
                                greedy_equivalency_labels,
                                summed_likelihood,
                                sampling_count,
                            )
                        ),
                    )
                ]

        metrics_dict = {}
        for cs in comparison_strategies:
            greedy_equivalency_labels = torch.cat(cs_q_labels[cs], dim=0)
            counting_p = np.concatenate(cs_us_counting[cs], axis=0)
            likelihood_p = np.concatenate(cs_us_likelihood[cs], axis=0)
            # q_p = torch.cat(cs_q_logits[cs], dim=0)[:, q_token_vec].softmax(dim=-1)

            acc = greedy_equivalency_labels.float().mean(dim=0)
            # q_pred = q_p.argmax(dim=-1)
            # q_acc = (q_pred == q_labels).float().mean(dim=0)

        ece_counting, _ = calibration(
            np.ones_like(greedy_equivalency_labels.detach().cpu().numpy()),
            greedy_equivalency_labels,
            counting_p,
        )

        ece_likelihood, _ = calibration(
            np.ones_like(greedy_equivalency_labels.detach().cpu().numpy()),
            greedy_equivalency_labels,
            likelihood_p,
        )

        metrics_dict.update(
            {
                "N": greedy_equivalency_labels.size(0),
                f"{cs}_acc": acc.item(),
                f"{cs}_ece_counting": ece_counting,
                f"{cs}_ece_likelihood": ece_likelihood,
            }
        )

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "sampling"), exist_ok=True)

        all_data["evals"] = {
            k: {qk: np.concatenate(qv, axis=0) for qk, qv in v.items()}
            for k, v in all_data["evals"].items()
        }

        pd.DataFrame(all_data["rows"]).to_csv(
            f"{log_dir}/rows_{accelerator.process_index}.csv", index=False
        )
        with open(os.path.join(log_dir, "sampling", f"q_{accelerator.process_index}.pt"), "wb") as f:
            torch.save(all_data["evals"], f)

        logging.debug(
            f"Data saved to {os.path.join(log_dir, 'sampling')} from process {accelerator.process_index}."
        )

    return metrics_dict


# # https://github.com/tonyzhaozh/few-shot-learning/blob/e04d8643be91c2cce63f33e07760ff75d5aa3ad0/run_extraction.py#L144C9-L144C9
# # Using the hack of contextual calibration for generation tasks from the original paper.
# # Calibrate first token of generation against first token of ground truth, then greedily decode the rest of the sequence BASED ON THE CALIBRATED TOKEN.
# # Therefore the first token is the one used for all ece computation/confidence, but the entire sequence matters for accuracy.
# @torch.inference_mode()
# def evaluate_contextual_calibration_oe(
#     accelerator,
#     model,
#     tokenizer,
#     loader,
#     prompt_style="choice",
#     query_format="roman_choice",
#     comparison_strategies=None,
#     max_new_tokens=30,
#     **_,
# ):
#     """
#     Assumes all answers are 1 token and end immediately with EOS token.
#     """

#     assert prompt_style == "oe"

#     device = accelerator.device
#     collate_fn = DataCollatorForSupervisedDataset(tokenizer)

#     if isinstance(model, PeftModel):
#         model.set_adapter("default")

#     all_y, all_logits = [], []
#     all_platt_logits = []

#     for raw_inputs in tqdm(loader, leave=False):
#         platt_logits = []

#         for cf_str in [
#             "Question: N/A",
#             "Question: ",
#             f"Question: {tokenizer.pad_token}",
#         ]:

#             calib_inputs = {
#                 **raw_inputs,
#                 "context": [cf_str] * len(raw_inputs["context"]),
#             }
#             # OE changes: prompt style
#             calib_inputs = prepare_batch(
#                 tokenizer, calib_inputs, prompt_style=prompt_style
#             )
#             # calib_inputs = prepare_batch(tokenizer, calib_inputs)
#             calib_inputs = collate_fn(calib_inputs)

#             calib_inputs = {k: v.to(device) for k, v in calib_inputs.items()}
#             calib_outputs = model(**calib_inputs)

#             # OE changes: extract oe inputs

#             calib_target_start_idx, _, _ = extract_oe_inputs(tokenizer, calib_inputs)
#             _c_logits = torch.squeeze(
#                 calib_outputs.logits[:, calib_target_start_idx, :], dim=1
#             )  # first token only

#             # _, _, _c_logits = extract_qa_exact(
#             #     tokenizer, calib_inputs, outputs=calib_outputs
#             # )

#             platt_logits.append(_c_logits)

#         ## Ensemble over context-free strings.
#         platt_logits = torch.stack(platt_logits).mean(dim=0)

#         # OE changes: prompt style
#         inputs = prepare_batch(tokenizer, raw_inputs, prompt_style=prompt_style)
#         inputs = collate_fn(inputs)

#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         outputs = model(**inputs)

#         # OE changes: extract first token
#         target_start_idx, _, oe_targets = extract_oe_inputs(tokenizer, inputs)
#         logits = torch.squeeze(outputs.logits[:, target_start_idx, :], dim=1)
#         y = oe_targets[:, 0]
#         # _, y, logits = extract_qa_exact(tokenizer, inputs, outputs=outputs)

#         [
#             l.append(v)
#             for l, v in zip(
#                 (all_y, all_logits, all_platt_logits),
#                 accelerator.gather_for_metrics((y, logits, platt_logits)),
#             )
#         ]

#     all_y, all_logits, all_platt_logits = [
#         torch.cat(l, dim=0) for l in (all_y, all_logits, all_platt_logits)
#     ]

#     all_p = all_logits.softmax(dim=-1)
#     all_y_hat = all_p.argmax(dim=-1)
#     acc = (all_y == all_y_hat).float().mean()
#     logits_ece, _ = calibration(
#         all_y, all_y_hat, all_p[torch.arange(all_p.size(0)), all_y_hat]
#     )

#     all_cal_logits = all_logits - all_platt_logits
#     all_cal_p = all_cal_logits.softmax(dim=-1)
#     all_cal_y_hat = all_cal_p.argmax(dim=-1)
#     cal_acc = (all_y == all_cal_y_hat).float().mean()
#     cal_ece, _ = calibration(
#         all_y, all_y_hat, all_cal_p[torch.arange(all_cal_p.size(0)), all_cal_y_hat]
#     )

#     return_dict = {
#         "N": all_y.size(0),
#         # OE Change: acc -> first_token_acc
#         "first_token_acc": acc.item(),
#         "logits_ece": logits_ece,
#         # OE Change: acc -> first token acc
#         "cal_first_token_acc": cal_acc.item(),
#         "cal_ece": cal_ece,
#     }

#     # CHANGE FOR OE
#     # generate the entire remainder of the sequence for each element

#     # all_oe_inputs = []
#     all_acc = {c: [] for c in comparison_strategies}
#     all_oe_target_strings, all_output_strings, all_question_strings = [], [], []

#     all_cal_p_index = 0

#     for inputs in tqdm(loader, leave=False):

#         inputs = prepare_batch(tokenizer, inputs, prompt_style=prompt_style)
#         inputs = collate_fn(inputs)

#         # extract first token
#         first_token = all_cal_p[
#             all_cal_p_index : all_cal_p_index + len(inputs["input_ids"])
#         ]
#         all_cal_p_index += len(inputs["input_ids"])

#         target_start_idx, oe_inputs_base, oe_targets = extract_oe_inputs(
#             tokenizer, inputs
#         )

#         oe_inputs = collate_fn(oe_inputs_base)
#         oe_inputs = {k: v.to(device) for k, v in oe_inputs.items()}

#         oe_inputs_extended = collate_fn(oe_inputs_base)
#         oe_inputs_extended = {k: v.to(device) for k, v in oe_inputs_extended.items()}

#         # add the calibrated first token
#         oe_inputs_extended["input_ids"] = torch.cat(
#             [
#                 oe_inputs_extended["input_ids"],
#                 first_token.argmax(dim=-1).unsqueeze(dim=1).to(device),
#             ],
#             dim=-1,
#         )
#         oe_inputs_extended["attention_mask"] = torch.cat(
#             [
#                 oe_inputs_extended["attention_mask"],
#                 torch.ones(oe_inputs_extended["attention_mask"].shape[0], 1).to(device),
#             ],
#             dim=-1,
#         )

#         # import pdb; pdb.set_trace()

#         if isinstance(model, PeftModel):
#             model.set_adapter("default")

#         # generate 30 - 1 more tokens
#         outputs = model.generate(
#             **oe_inputs_extended, max_new_tokens=max_new_tokens - 1
#         )

#         # convert those new tokens to the generated strings
#         output_strings = tokenizer.batch_decode(
#             outputs[..., target_start_idx:],
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False,
#         )

#         # these are the ground truth strings for this batch
#         oe_target_strings = tokenizer.batch_decode(
#             oe_targets, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )

#         question_strings = tokenizer.batch_decode(
#             oe_inputs["input_ids"],
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False,
#         )

#         # prepare the calibration query with open ended text
#         # the calculation of the accuracy is done within this function
#         for c in comparison_strategies:
#             _, _, acc = prepare_oe_calibration_query(
#                 tokenizer,
#                 oe_target_strings,
#                 output_strings,
#                 question_strings,
#                 format=query_format,
#                 comparison_strategy=c,
#             )

#             acc = acc.to(device)

#             [
#                 l.append(v)
#                 for l, v in zip(
#                     (all_acc[c]),
#                     accelerator.gather_for_metrics((acc)),
#                 )
#             ]

#         [
#             l.append(v)
#             for l, v in zip(
#                 (all_oe_target_strings, all_output_strings, all_question_strings),
#                 accelerator.gather_for_metrics(
#                     (oe_target_strings, output_strings, question_strings)
#                 ),
#             )
#         ]

#     for c in comparison_strategies:
#         all_acc_c = np.array([e.cpu().numpy() for e in all_acc[c]]).flatten()
#         return_dict[f"{c}_acc"] = np.sum(all_acc_c) / len(all_acc_c)

#         c_ece, _ = calibration(
#             np.ones_like(all_acc_c),
#             all_acc_c,
#             all_cal_p[torch.arange(all_cal_p.size(0)), all_cal_y_hat].cpu().numpy(),
#         )

#         return_dict[f"{c}_ece"] = c_ece

#     return return_dict


VERBAL_ELICITATION_UNC_QUERIES = {
    "2s1CoT": "".join(
        [
            "Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\n",
            "For example:\n\n",
            "Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n",
        ]
    ),
    "2s1g": "".join(
        [
            "Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\n",
            "For example:\n\n",
            "Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n",
        ]
    ),
    "2s2g": "".join(
        [
            "Provide the probability that each of your guesses is correct. Give ONLY the probabilities, no other words or explanation.\n\n",
            "For example:\n\n",
            "P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n",
            "P2: <the probability between 0.0 and 1.0 that G2 is correct, without any extra commentary whatsoever; just the probability!>\n",
        ]
    ),
    "2s4g": "".join(
        [
            "Provide the probability that each of your guesses is correct. Give ONLY the probabilities, no other words or explanation.\n\n",
            "For example:\n\n",
            "P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n",
            "P2: <the probability between 0.0 and 1.0 that G2 is correct, without any extra commentary whatsoever; just the probability!>\n",
            "P3: <the probability between 0.0 and 1.0 that G3 is correct, without any extra commentary whatsoever; just the probability!>\n",
            "P4: <the probability between 0.0 and 1.0 that G4 is correct, without any extra commentary whatsoever; just the probability!>\n",
        ]
    ),
}


def parse_verbal_elicitation_oe(
    output_string,
    ve_style,
    stage=1,
):
    stages, guess_style = ve_style.split("s")
    stages = int(stages)
    num_guesses = int(guess_style[0])
    is_CoT = guess_style[1] == "C"

    if stages == 1:
        expected_lines = 2 * num_guesses
    else:
        expected_lines = num_guesses
        if stage == 1:
            expected_lines += 1 if is_CoT else 0

    output_string = output_string.replace("\n\n", "\n")
    parts = output_string.split("\n")[:expected_lines]

    if len(parts) == 0:
        return "", torch.tensor([0]), torch.tensor([[1, 0]])

    def parse_guess(guess, idx):
        if (num_guesses == 1) and ("Probability:" in guess):
            return None
        elif (num_guesses > 1) and any(
            [(f"P{i+1}:" in guess) for i in range(num_guesses)]
        ):
            return None

        guess = guess.split(":")[-1].strip()
        return guess

    def parse_prob(prob, idx):
        if (num_guesses == 1) and ("Probability:" not in prob):
            return None
        elif (num_guesses > 1) and (f"P{idx+1}:" not in prob):
            return None
        try:
            prob = float(prob.split(":")[-1].strip())
        except:
            prob = 0.5
        return prob

    if stages == 1:
        guesses = [parse_guess(g, i) for i, g in enumerate(parts[::2])]
        probs = [parse_prob(p, i) for i, p in enumerate(parts[1::2])]
    elif stage == 1:
        guesses = [parse_guess(g, i) for i, g in enumerate(parts)]
        probs = []
    elif stage == 2:
        guesses = []
        probs = [parse_prob(p, i) for i, p in enumerate(parts)]

    if guesses[0] is None:
        return "", torch.tensor([0]), torch.tensor([[1, 0]])
    elif any([g is None for g in guesses]):
        output = guesses[0]
        prob = 0.5 if (probs[0] is None) else probs[0]
    elif all([p is None for p in probs]):
        output = guesses[0]
        prob = 0.5
    else:
        guesses, probs = zip(*[(g, p) for g, p in zip(guesses, probs) if p is not None])
        max_idx = np.argmax(probs)
        output = guesses[max_idx]
        prob = probs[max_idx]

    y = torch.tensor([prob > 0.5])
    logits = torch.tensor([[1 - prob, prob]])

    return output, y, logits


# @torch.inference_mode()
# def evaluate_verbal_elicitation_oe(
#     ve_style,
#     accelerator,
#     model,
#     tokenizer,
#     loader,
#     prompt_style="oe",
#     comparison_strategies=None,
#     max_new_tokens=30,
#     output_row_path=None,
#     **_,
# ):
#     assert prompt_style == "oe"
#     assert (not comparison_strategies is None) and len(comparison_strategies) > 0

#     device = accelerator.device

#     stages, guess_style = ve_style.split("s")
#     stages = int(stages)
#     num_guesses = int(guess_style[0])
#     max_new_tokens = max(30 * num_guesses, max_new_tokens)

#     generation_config = GenerationConfig(
#         pad_token_id=tokenizer.pad_token_id,
#         bos_token_id=tokenizer.bos_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         max_new_tokens=max_new_tokens,
#     )

#     all_unc_y, all_unc_logits = {c: [] for c in comparison_strategies}, {
#         c: [] for c in comparison_strategies
#     }
#     all_acc = {c: [] for c in comparison_strategies}
#     all_oe_target_strings, all_output_strings, all_question_strings = [], [], []

#     output_generator = generate_output(
#         accelerator,
#         model,
#         tokenizer,
#         loader,
#         prompt_style=prompt_style,
#         generation_config=generation_config,
#     )

#     # i = 0
#     # for example in output_generator:
#     #     from pprint import pprint
#     #     pprint(example)

#     #     parse_verbal_elicitation_oe(
#     #         example["output"], ve_style, stage=1
#     #     )
#     #     # print("\n")
#     #     i += 1
#     #     if i > 10:
#     #         break

#     # print(1/0)

#     for example in output_generator:
#         output, raw_input, target = (
#             example["output"],
#             example["raw_input"],
#             example["target"],
#         )

#         output, unc_y, unc_logits = parse_verbal_elicitation_oe(
#             output, ve_style, stage=1
#         )

#         unc_y = unc_y.to(device)
#         unc_logits = unc_logits.to(device)

#         input_question_string = example["context"]
#         oe_target_strings = [target]
#         output_strings = [output]
#         question_strings = [input_question_string]

#         for c in comparison_strategies:
#             acc = torch.tensor(
#                 grade_oe_preds(
#                     oe_target_strings,
#                     output_strings,
#                     question_strings,
#                     comparison_strategy=c,
#                 )
#             ).to(device)

#             [
#                 l.append(v)
#                 for l, v in zip(
#                     (all_unc_y[c], all_unc_logits[c], all_acc[c]),
#                     (unc_y, unc_logits, acc),
#                 )
#             ]

#         [
#             l.append(v)
#             for l, v in zip(
#                 (all_oe_target_strings, all_output_strings, all_question_strings),
#                 (oe_target_strings, output_strings, question_strings),
#             )
#         ]

#     all_oe_target_strings, all_output_strings, all_question_strings = (
#         sum(all_oe_target_strings, []),
#         sum(all_output_strings, []),
#         sum(all_question_strings, []),
#     )

#     return_dict = {}
#     # note if using multiple gpus/processes, these string lists will be incomplete.
#     dump = {
#         "oe_target_strings": all_oe_target_strings,
#         "output_strings": all_output_strings,
#         "question_strings": all_question_strings,
#     }

#     for c in comparison_strategies:
#         all_unc_y_c, all_unc_p, all_acc_c = [
#             torch.cat(l, dim=0) for l in (all_unc_y[c], all_unc_logits[c], all_acc[c])
#         ]

#         acc = all_acc_c.float().mean()

#         all_unc_y_c = (
#             (all_unc_y_c.unsqueeze(-1) == torch.arange(2).to(device))
#             .long()
#             .argmax(dim=-1)
#         )
#         all_unc_y_hat = all_unc_p.argmax(dim=-1)
#         unc_acc = (all_unc_y_c == all_unc_y_hat).float().mean()
#         unc_ece, _ = calibration(
#             all_unc_y_c,
#             all_unc_y_hat,
#             all_unc_p[torch.arange(all_unc_p.size(0)), all_unc_y_hat],
#         )

#         return_dict.update(
#             {
#                 f"{c}_acc": acc.item(),
#                 f"{c}_unc_acc": unc_acc.item(),
#                 f"{c}_unc_ece": unc_ece,
#                 "N": all_unc_p.size(0),
#             }
#         )

#         dump.update(
#             {
#                 f"{c}_acc": all_acc_c.cpu().numpy(),
#                 f"{c}_all_unc_y": all_unc_y_c.cpu().numpy(),
#                 f"{c}_all_unc_y_hat": all_unc_y_hat.cpu().numpy(),
#                 f"{c}_unc_acc": unc_acc.item(),
#                 f"{c}_unc_ece": unc_ece,
#             }
#         )

#     if accelerator.num_processes == 1 and output_row_path is not None:
#         # create parent dir if it doesn't exist
#         import os

#         os.makedirs(os.path.dirname(output_row_path), exist_ok=True)
#         pd.DataFrame(dump).to_csv(output_row_path, escapechar="\\")

#     return return_dict
