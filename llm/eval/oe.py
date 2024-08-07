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

from ..random import FixedSeed
from ..datasets import LMText, LabeledStringDataCollator, prepare_uncertainty_query
from ..datasets.llm_utils_oe import (
    equivalency_grading,
    sanitize_generations,
)
from .third_party.calibration import calibration


@torch.inference_mode()
def evaluate_uncertainty_sampling_oe(
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
                _, greedy_equivalency_labels, _ = prepare_uncertainty_query(
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

            acc = greedy_equivalency_labels.float().mean(dim=0)

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
        with open(
            os.path.join(log_dir, "sampling", f"q_{accelerator.process_index}.pt"), "wb"
        ) as f:
            torch.save(all_data["evals"], f)

        logging.debug(
            f"Data saved to {os.path.join(log_dir, 'sampling')} from process {accelerator.process_index}."
        )

    return metrics_dict


VERBAL_ELICITATION_UNC_QUERIES = (
    "Provide the probability that your answer is correct. "
    "Give ONLY the probability, no other words or explanation.\n\n"
    "For example:\n\n"
    "Probability: <the probability between 0.0 and 1.0 that your guess is correct, "
    "without any extra commentary whatsoever; just the probability!>\n\n"
    "Include probability for the answer below:\n"
    "Probability:"
)


def parse_verbal_elicitation_oe(
    output_string,
):
    output_string = output_string.replace("\n\n", "\n")
    output_string = output_string.replace(":\n", ":")
    output_string = output_string.strip("\n").split("\n")[0]

    try:
        prob = float(output_string.split(":")[-1].strip())
    except:
        prob = 0.5

    y = torch.tensor([prob > 0.5])
    logits = torch.tensor([[1 - prob, prob]]).log()

    return y, logits


@torch.inference_mode()
def evaluate_verbal_elicitation_oe(
    ve_style,
    accelerator,
    model,
    tokenizer,
    loader,
    prompt_style="oe",
    comparison_strategies=None,
    max_new_tokens=30,
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

    collate_fn = LabeledStringDataCollator(tokenizer)

    cs_q_labels = {c: [] for c in comparison_strategies}
    cs_q_logits = {c: [] for c in comparison_strategies}

    all_data = {
        "rows": [],
        "evals": {c: {"q_labels": [], "q_logits": []} for c in comparison_strategies},
    }

    for inputs in tqdm(loader):
        inputs.pop("embedding", None)
        inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
        targets = [inp.pop("target") for inp in inputs]

        if "output" in inputs[0]:
            generations = [inp.pop("output") for inp in inputs]
        else:
            pass

        all_data["rows"].extend(
            [
                {**inp, "target": tgt, "output": out}
                for inp, tgt, out in zip(inputs, targets, generations)
            ]
        )

        for cs in comparison_strategies:
            q_labels = (
                [inp.pop("query_label").item() for inp in inputs]
                if "query_label" in inputs[0]
                else None
            )
            q_labels = torch.Tensor(q_labels).to(accelerator.device)

            uncertainty_prompt = VERBAL_ELICITATION_UNC_QUERIES

            contexts = [str(LMText.from_(inp)) for inp in inputs]

            q_inputs = [
                {
                    "context": f"{c + ' ' + p.strip()}\n\n",
                    "target_prompt": uncertainty_prompt,
                }
                for c, p in zip(contexts, generations)
            ]

            gen_inputs = {
                k: v.to(accelerator.device) for k, v in collate_fn(q_inputs).items()
            }

            gen_output = model.generate(
                **gen_inputs, generation_config=generation_config
            )

            gen_output = [
                o[i.size(0) :] for o, i in zip(gen_output, gen_inputs["input_ids"])
            ]

            gen_output = tokenizer.batch_decode(
                gen_output,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            q_logits = torch.cat(
                [parse_verbal_elicitation_oe(x)[1] for x in gen_output]
            ).to(accelerator.device)

            all_data["evals"][cs]["q_labels"].append(q_labels.detach())
            all_data["evals"][cs]["q_logits"].append(q_logits.detach())

            [
                l.append(v.cpu())
                for l, v in zip(
                    (cs_q_labels[cs], cs_q_logits[cs]),
                    accelerator.gather_for_metrics((q_labels, q_logits)),
                )
            ]

    metrics_dict = {}
    for cs in comparison_strategies:
        q_labels = torch.cat(cs_q_labels[cs], dim=0)
        q_p = torch.cat(cs_q_logits[cs], dim=0).softmax(dim=-1)

        acc = q_labels.float().mean(dim=0)
        q_pred = q_p.argmax(dim=-1)
        q_acc = (q_pred == q_labels).float().mean(dim=0)

        q_ece, _ = calibration(
            q_labels,
            q_pred,
            q_p[torch.arange(q_p.size(0)), q_pred].float(),
        )

        try:
            q_auroc = roc_auc_score(
                q_labels.cpu(),
                q_p[torch.arange(q_p.size(0)), 1].float().cpu(),
            )
        except ValueError:
            q_auroc = float("nan")
            logging.exception("AUROC calculation failed.", exc_info=True)

        metrics_dict.update(
            {
                "N": q_labels.size(0),
                f"{cs}_acc": acc.item(),
                f"{cs}_unc_acc": q_acc.item(),
                f"{cs}_unc_auroc": q_auroc,
                f"{cs}_unc_ece": q_ece,
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


# VERBAL_ELICITATION_UNC_QUERIES = {
#     "2s1CoT": "".join(
#         [
#             "Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\n",
#             "For example:\n\n",
#             "Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\n",
#             "Include probability of the guess below:\n",
#             "Probability:",
#         ]
#     ),
#     "2s1g": "".join(
#         [
#             "Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\n",
#             "For example:\n\n",
#             "Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\n",
#             "Include probability of each guess below:\n",
#             "Probability:",
#         ]
#     ),
#     "2s2g": "".join(
#         [
#             "Provide the probability that each of your guesses is correct. Give ONLY the probabilities, no other words or explanation.\n\n",
#             "For example:\n\n",
#             "P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n",
#             "P2: <the probability between 0.0 and 1.0 that G2 is correct, without any extra commentary whatsoever; just the probability!>\n\n",
#             "Include probability of the guess below:\n\n",
#             "P1:",
#         ]
#     ),
#     "2s4g": "".join(
#         [
#             "Provide the probability that each of your guesses is correct. Give ONLY the probabilities, no other words or explanation.\n\n",
#             "For example:\n\n",
#             "P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n",
#             "P2: <the probability between 0.0 and 1.0 that G2 is correct, without any extra commentary whatsoever; just the probability!>\n",
#             "P3: <the probability between 0.0 and 1.0 that G3 is correct, without any extra commentary whatsoever; just the probability!>\n",
#             "P4: <the probability between 0.0 and 1.0 that G4 is correct, without any extra commentary whatsoever; just the probability!>\n\n",
#             "Include probability of the guess below:\n\n",
#             "P1:",
#         ]
#     ),
# }


# def parse_verbal_elicitation_oe(
#     output_string,
#     ve_style,
#     stage=1,
#     prev_guesses=None,
# ):
#     stages, guess_style = ve_style.split("s")
#     stages = int(stages)
#     num_guesses = int(guess_style[0])
#     is_CoT = guess_style[1] == "C"

#     if stages == 1:
#         expected_lines = 2 * num_guesses
#     else:
#         expected_lines = num_guesses
#         if stage == 1:
#             expected_lines += 1 if is_CoT else 0

#     output_string = output_string.replace("\n\n", "\n")
#     output_string = output_string.replace(":\n", ":")
#     parts = output_string.strip("\n").split("\n")[:expected_lines]

#     if is_CoT and (stage == 1):
#         explanation = parts[0]
#         parts = parts[1:]

#     if len(parts) == 0:
#         return "", torch.tensor([0]), torch.tensor([[1, 0]])

#     def parse_guess(guess, idx):
#         if (num_guesses == 1) and ("Probability:" in guess):
#             return None
#         elif (num_guesses > 1) and any(
#             [(f"P{i+1}:" in guess) for i in range(num_guesses)]
#         ):
#             return None

#         guess = guess.split(":")[-1].strip()
#         return guess

#     def parse_prob(prob, idx):
#         if (num_guesses == 1) and (stage == 1) and ("Probability:" not in prob):
#             return None
#         elif num_guesses > 1:
#             if (stage == 1) and (f"P{idx+1}:" not in prob):
#                 return None
#             elif (stage == 2) and (idx > 1) and (f"P{idx+1}:" in prob):
#                 return None
#         try:
#             prob = float(prob.split(":")[-1].strip())
#         except:
#             prob = 0.5
#         return prob

#     if stages == 1:
#         guesses = [parse_guess(g, i) for i, g in enumerate(parts[::2])]
#         probs = [parse_prob(p, i) for i, p in enumerate(parts[1::2])]
#     elif stage == 1:
#         guesses = [parse_guess(g, i) for i, g in enumerate(parts)]
#         if is_CoT:
#             return (guesses, explanation), None, None
#         else:
#             return guesses, None, None
#     elif stage == 2:
#         assert prev_guesses is not None
#         probs = [parse_prob(p, i) for i, p in enumerate(parts)]
#         guesses = prev_guesses

#     if guesses[0] is None:
#         return "", torch.tensor([0]), torch.tensor([[1, 0]])
#     elif any([g is None for g in guesses]):
#         output = guesses[0]
#         prob = 0.5 if (probs[0] is None) else probs[0]
#     elif all([p is None for p in probs]):
#         output = guesses[0]
#         prob = 0.5
#     else:
#         guesses, probs = zip(*[(g, p) for g, p in zip(guesses, probs) if p is not None])
#         max_idx = np.argmax(probs)
#         output = guesses[max_idx]
#         prob = probs[max_idx]

#     y = torch.tensor([prob > 0.5])
#     logits = torch.tensor([[1 - prob, prob]])

#     return output, y, logits

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
#     collate_fn = DataCollatorForSupervisedDataset(tokenizer)

#     stages, guess_style = ve_style.split("s")
#     stages = int(stages)
#     num_guesses = int(guess_style[0])
#     is_CoT = guess_style[1] == "C"
#     if is_CoT:
#         max_new_tokens = 200
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

#     for example in output_generator:
#         output, raw_input, target = (
#             example["output"],
#             example["raw_input"],
#             example["target"],
#         )

#         output, unc_y, unc_logits = parse_verbal_elicitation_oe(
#             output, ve_style, stage=1
#         )

#         # 2 STAGE CASE
#         if unc_logits is None:

#             uncertainty_prompt = VERBAL_ELICITATION_UNC_QUERIES[ve_style]

#             guesses_str = ""
#             if is_CoT:
#                 output, explanation = output
#                 guesses_str = f"{explanation}\n\n"

#             if num_guesses == 1:
#                 guesses_str = guesses_str + f"Guess: {output[0]}"
#             else:
#                 guesses_str = guesses_str + "\n\n".join(
#                     [f"G{i+1}: {g}" for i, g in enumerate(output)]
#                 )

#             sample = (
#                 example["prompt"]
#                 + example["context"]
#                 + example["target_prompt"]
#                 + guesses_str
#                 + "\n\n"
#                 + uncertainty_prompt
#             )

#             tokenizer_args = dict(
#                 padding="longest",
#                 truncation=True,
#             )
#             tokenize_dict = tokenizer(str(sample), **tokenizer_args)

#             gen_inputs = {
#                 k: v.to(device) for k, v in collate_fn([tokenize_dict]).items()
#             }

#             gen_config = GenerationConfig(
#                 pad_token_id=tokenizer.pad_token_id,
#                 bos_token_id=tokenizer.bos_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#                 max_new_tokens=max_new_tokens,
#             )

#             gen_output = model.generate(**gen_inputs, generation_config=gen_config)[0]

#             gen_output = tokenizer.decode(
#                 gen_output[gen_inputs.get("input_ids")[0].size(0) :],
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=False,
#             )

#             output, unc_y, unc_logits = parse_verbal_elicitation_oe(
#                 gen_output, ve_style, stage=2, prev_guesses=output
#             )

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
#                 f"{c}_all_unc_p": all_unc_p[..., 1].cpu().numpy(),
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
