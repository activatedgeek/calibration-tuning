from tqdm.auto import tqdm
import torch
from peft import PeftModel
import pandas as pd
import numpy as np

from llm.datasets.llm_utils import (
    get_token_vec,
    DataCollatorForSupervisedDataset,
    extract_qa_exact,
    prepare_batch,
)
from llm.datasets.llm_utils_oe import (
    extract_oe_inputs,
    prepare_oe_calibration_query,
    grade_oe_preds,
    clustering_equivalency_with_oracle,
    openai_query,
)
from llm.eval.third_party.calibration import calibration
from llm.utils.generate_utils import generate_output
from transformers import GenerationConfig


@torch.inference_mode()
def evaluate_oe(
    accelerator,
    model,
    tokenizer,
    loader,
    prompt_style="oe",
    query_format="roman_choice",
    comparison_strategies=None,
    max_new_tokens=30,
    output_row_path=None,
):
    assert prompt_style == "oe"
    assert (not comparison_strategies is None) and len(comparison_strategies) > 0

    device = accelerator.device
    collate_fn = DataCollatorForSupervisedDataset(tokenizer)

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )

    query_token_vec = get_token_vec(tokenizer, format=query_format)
    all_unc_y, all_unc_logits = {c: [] for c in comparison_strategies}, {
        c: [] for c in comparison_strategies
    }
    all_acc = {c: [] for c in comparison_strategies}
    all_oe_target_strings, all_output_strings, all_question_strings = [], [], []

    output_generator = generate_output(
        accelerator,
        model,
        tokenizer,
        loader,
        prompt_style=prompt_style,
        generation_config=generation_config,
    )

    for example in output_generator:
        output, raw_input, target = (
            example["output"],
            example["raw_input"],
            example["target"],
        )
        input_question_string = example["context"]
        oe_target_strings = [target]
        output_strings = [output]
        question_strings = [input_question_string]

        for c in comparison_strategies:
            query_inputs, acc = prepare_oe_calibration_query(
                tokenizer,
                oe_target_strings,
                output_strings,
                question_strings,
                format=query_format,
                comparison_strategy=c,
            )

            acc = acc.to(device)
            query_inputs = collate_fn(query_inputs)

            if isinstance(model, PeftModel) and "query" in model.peft_config:
                model.set_adapter("query")

            query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
            query_outputs = model(**query_inputs)

            _, unc_y, unc_logits = extract_qa_exact(
                tokenizer, query_inputs, outputs=query_outputs
            )

            [
                l.append(v)
                for l, v in zip(
                    (all_unc_y[c], all_unc_logits[c], all_acc[c]),
                    accelerator.gather_for_metrics((unc_y, unc_logits, acc)),
                )
            ]

        [
            l.append(v)
            for l, v in zip(
                (all_oe_target_strings, all_output_strings, all_question_strings),
                accelerator.gather_for_metrics(
                    (oe_target_strings, output_strings, question_strings)
                ),
            )
        ]

    all_oe_target_strings, all_output_strings, all_question_strings = (
        sum(all_oe_target_strings, []),
        sum(all_output_strings, []),
        sum(all_question_strings, []),
    )

    return_dict = {}
    # note if using multiple gpus/processes, these string lists will be incomplete.
    dump = {
        "oe_target_strings": all_oe_target_strings,
        "output_strings": all_output_strings,
        "question_strings": all_question_strings,
    }

    for c in comparison_strategies:

        query_token_vec = query_token_vec.to(device)
        all_unc_y_c, all_unc_logits_c, all_acc_c = [
            torch.cat(l, dim=0) for l in (all_unc_y[c], all_unc_logits[c], all_acc[c])
        ]

        acc = all_acc_c.float().mean()

        all_unc_y_c, all_unc_p = (
            (all_unc_y_c.unsqueeze(-1) == query_token_vec).long().argmax(dim=-1),
            all_unc_logits_c[:, query_token_vec].softmax(dim=-1),
        )
        all_unc_y_hat = all_unc_p.argmax(dim=-1)
        unc_acc = (all_unc_y_c == all_unc_y_hat).float().mean()
        unc_ece, _ = calibration(
            all_unc_y_c,
            all_unc_y_hat,
            all_unc_p[torch.arange(all_unc_p.size(0)), all_unc_y_hat],
        )

        return_dict.update(
            {
                f"{c}_acc": acc.item(),
                f"{c}_unc_acc": unc_acc.item(),
                f"{c}_unc_ece": unc_ece,
                "N": all_unc_p.size(0),
            }
        )

        dump.update(
            {
                f"{c}_acc": all_acc_c.cpu().numpy(),
                f"{c}_all_unc_y": all_unc_y_c.cpu().numpy(),
                f"{c}_all_unc_y_hat": all_unc_y_hat.cpu().numpy(),
                f"{c}_unc_acc": unc_acc.item(),
                f"{c}_unc_ece": unc_ece,
            }
        )

    if accelerator.num_processes == 1 and output_row_path is not None:
        pd.DataFrame(dump).to_csv(output_row_path, escapechar="\\", index=False)

    return return_dict


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
    output_row_path=None,
    top_p=0.95,
    k=10,
):
    assert prompt_style == "oe"
    assert (not comparison_strategies is None) and len(comparison_strategies) > 0

    device = accelerator.device
    collate_fn = DataCollatorForSupervisedDataset(tokenizer)

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )

    generation_config_sampling = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
    )

    query_token_vec = get_token_vec(tokenizer, format=query_format)
    all_prob = {c: [] for c in comparison_strategies}
    all_likelihood_accumulate = {c: [] for c in comparison_strategies}
    all_normalized_likelihood_accumulate = {c: [] for c in comparison_strategies}
    all_likelihood_outputs = []
    all_acc = {c: [] for c in comparison_strategies}
    (
        all_oe_target_strings,
        all_output_strings,
        all_question_strings,
        all_generations,
        all_match_scores,
    ) = ([], [], [], [], [])

    output_generator = generate_output(
        accelerator,
        model,
        tokenizer,
        loader,
        prompt_style=prompt_style,
        generation_config=generation_config,
        generation_config_sampling=generation_config_sampling,
        k=k,
    )

    for example in output_generator:
        output, raw_input, target = (
            example["output"],
            example["raw_input"],
            example["target"],
        )
        input_question_string = example["context"]
        oe_target_strings = [target]
        output_strings = [output]
        question_strings = [input_question_string]

        # generate 30 more tokens k addtl times for the uncertainty estimation
        sampled_outputs = example["sampled_outputs"]
        sampled_probs = example["sampled_outputs"]
        outputs_list = []
        length_normalized_likelihoods_list = []
        for i in range(k):

            length_normalized_likelihoods_list.append(
                [
                    np.exp(
                        np.sum(
                            np.log(
                                outputs_pre_cluster[..., target_start_idx:]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        )
                        / len(outputs_pre_cluster[..., target_start_idx:][0])
                    )
                ]
            )

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
        prob = []
        match_scores = []
        likelihood_score = []
        normalized_likelihood_score = []
        likelihood_output_list = []
        for i, question_string, greedy in zip(
            range(len(question_strings)), question_strings, output_strings
        ):
            n_cluster = 0
            likelihood_accumulate = 0

            for sample, likelihoods in zip(
                outputs_list, length_normalized_likelihoods_list
            ):
                likelihood = likelihoods[i]
                generation = sample[i]

                add_to_cluster = clustering_equivalency_with_oracle(
                    greedy,
                    generation,
                    question_strings[0],
                    oracle_fn=openai_query,
                    oracle_kwargs={"openai_model_name": "gpt-3.5-turbo-1106"},
                )

                if add_to_cluster:
                    n_cluster += 1
                    likelihood_accumulate += likelihood

                match_scores.append(add_to_cluster)
            # then prob = (size of the cluster associated with greedy) / k
            prob.append(n_cluster / k)
            likelihood_score.append(likelihood_accumulate)
            normalized_likelihood_score.append(
                likelihood_accumulate
                / sum([l[i] for l in length_normalized_likelihoods_list])
            )
            likelihood_output_list.append(
                [l[i] for l in length_normalized_likelihoods_list]
            )

        prob = np.array(prob)
        likelihood_score = np.array(likelihood_score)
        normalized_likelihood_score = np.array(normalized_likelihood_score)

        # prob = outputs_list_length_normalized[argmax_i] / np.sum(outputs_list_length_normalized)
        # outputs_argmax = outputs_list[argmax_i].unsqueeze(dim=0)
        # outputs = model.generate(**oe_inputs, max_new_tokens=max_new_tokens)

        # prepare the calibration query with open ended text
        # the calculation of the accuracy is done within this function
        for c in comparison_strategies:
            _, _, acc = prepare_oe_calibration_query(
                tokenizer,
                oe_target_strings,
                output_strings,
                question_strings,
                format=query_format,
                comparison_strategy=c,
            )

            acc = acc.to(device)

            [
                l.append(v)
                for l, v in zip(
                    (
                        all_prob[c],
                        all_likelihood_accumulate[c],
                        all_normalized_likelihood_accumulate[c],
                        all_acc[c],
                    ),
                    accelerator.gather_for_metrics(
                        (prob, likelihood_score, normalized_likelihood_score, acc)
                    ),
                )
            ]

        [
            l.append(v)
            for l, v in zip(
                (
                    all_oe_target_strings,
                    all_output_strings,
                    all_question_strings,
                    all_generations,
                    all_match_scores,
                    all_likelihood_outputs,
                ),
                accelerator.gather_for_metrics(
                    (
                        oe_target_strings,
                        output_strings,
                        question_strings,
                        sum(outputs_list, []),
                        match_scores,
                        likelihood_output_list,
                    )
                ),
            )
        ]

    return_dict = {}

    all_oe_target_strings, all_output_strings, all_question_strings = (
        sum(all_oe_target_strings, []),
        sum(all_output_strings, []),
        sum(all_question_strings, []),
    )
    all_likelihood_outputs = sum(all_likelihood_outputs, [])
    dump = {
        "oe_target_strings": all_oe_target_strings,
        "output_strings": all_output_strings,
        "question_strings": all_question_strings,
        "all_generations": all_generations,
        "all_match_scores": all_match_scores,
        "all_likelihood_outputs": all_likelihood_outputs,
    }

    for c in comparison_strategies:

        all_acc_c = np.concatenate([c.cpu() for c in all_acc[c]], axis=0)
        all_prob_c = np.concatenate([p for p in all_prob[c]], axis=0)
        all_likelihood_accumulate_c = np.concatenate(
            [l for l in all_likelihood_accumulate[c]], axis=0
        )
        all_normalized_likelihood_accumulate_c = np.concatenate(
            [l for l in all_normalized_likelihood_accumulate[c]], axis=0
        )

        acc = all_acc_c.mean()
        # import pdb; pdb.set_trace()
        assert len(all_prob_c) == len(all_acc_c)
        assert len(all_likelihood_accumulate_c) == len(all_acc_c)
        ece, _ = calibration(
            np.ones_like(all_prob_c),
            all_acc_c.astype(dtype=np.dtype("i4")),
            np.array(all_prob_c),
        )

        likelihood_ece, _ = calibration(
            np.ones_like(all_likelihood_accumulate_c),
            all_acc_c.astype(dtype=np.dtype("i4")),
            np.array(all_likelihood_accumulate_c),
        )

        likelihood_normalized, _ = calibration(
            np.ones_like(all_normalized_likelihood_accumulate_c),
            all_acc_c.astype(dtype=np.dtype("i4")),
            np.array(all_normalized_likelihood_accumulate_c),
        )

        return_dict.update(
            {
                f"{c}_acc": acc.item(),
                f"{c}_ece_counting": ece,
                f"{c}_ece_likelihood": likelihood_ece,
                f"{c}_ece_likelihood_normalized": likelihood_normalized,
                "N": all_unc_p.size(0),
            }
        )

        dump.update(
            {
                f"{c}_acc": all_acc_c,
                f"{c}_counting_prob": all_prob_c,
                f"{c}_likelihood_accumulate": all_likelihood_accumulate_c,
                f"{c}_likelihood_normalized": all_normalized_likelihood_accumulate_c,
                # f"{c}_all_unc_y": all_unc_y_c.cpu().numpy(),
                # f"{c}_all_unc_y_hat": all_unc_y_hat.cpu().numpy(),
            }
        )

    if accelerator.num_processes == 1 and output_row_path is not None:
        pd.DataFrame(dump).to_csv(output_row_path, escapechar="\\", index=False)

    return return_dict


# https://github.com/tonyzhaozh/few-shot-learning/blob/e04d8643be91c2cce63f33e07760ff75d5aa3ad0/run_extraction.py#L144C9-L144C9
# Using the hack of contextual calibration for generation tasks from the original paper.
# Calibrate first token of generation against first token of ground truth, then greedily decode the rest of the sequence BASED ON THE CALIBRATED TOKEN.
# Therefore the first token is the one used for all ece computation/confidence, but the entire sequence matters for accuracy.
@torch.inference_mode()
def evaluate_contextual_calibration_oe(
    accelerator,
    model,
    tokenizer,
    loader,
    prompt_style="choice",
    query_format="roman_choice",
    comparison_strategies=None,
    max_new_tokens=30,
    **_,
):
    """
    Assumes all answers are 1 token and end immediately with EOS token.
    """

    assert prompt_style == "oe"

    device = accelerator.device
    collate_fn = DataCollatorForSupervisedDataset(tokenizer)

    if isinstance(model, PeftModel):
        model.set_adapter("default")

    all_y, all_logits = [], []
    all_platt_logits = []

    for raw_inputs in tqdm(loader, leave=False):
        platt_logits = []

        for cf_str in [
            "Question: N/A",
            "Question: ",
            f"Question: {tokenizer.pad_token}",
        ]:

            calib_inputs = {
                **raw_inputs,
                "context": [cf_str] * len(raw_inputs["context"]),
            }
            # OE changes: prompt style
            calib_inputs = prepare_batch(
                tokenizer, calib_inputs, prompt_style=prompt_style
            )
            # calib_inputs = prepare_batch(tokenizer, calib_inputs)
            calib_inputs = collate_fn(calib_inputs)

            calib_inputs = {k: v.to(device) for k, v in calib_inputs.items()}
            calib_outputs = model(**calib_inputs)

            # OE changes: extract oe inputs

            calib_target_start_idx, _, _ = extract_oe_inputs(tokenizer, calib_inputs)
            _c_logits = torch.squeeze(
                calib_outputs.logits[:, calib_target_start_idx, :], dim=1
            )  # first token only

            # _, _, _c_logits = extract_qa_exact(
            #     tokenizer, calib_inputs, outputs=calib_outputs
            # )

            platt_logits.append(_c_logits)

        ## Ensemble over context-free strings.
        platt_logits = torch.stack(platt_logits).mean(dim=0)

        # OE changes: prompt style
        inputs = prepare_batch(tokenizer, raw_inputs, prompt_style=prompt_style)
        inputs = collate_fn(inputs)

        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        # OE changes: extract first token
        target_start_idx, _, oe_targets = extract_oe_inputs(tokenizer, inputs)
        logits = torch.squeeze(outputs.logits[:, target_start_idx, :], dim=1)
        y = oe_targets[:, 0]
        # _, y, logits = extract_qa_exact(tokenizer, inputs, outputs=outputs)

        [
            l.append(v)
            for l, v in zip(
                (all_y, all_logits, all_platt_logits),
                accelerator.gather_for_metrics((y, logits, platt_logits)),
            )
        ]

    all_y, all_logits, all_platt_logits = [
        torch.cat(l, dim=0) for l in (all_y, all_logits, all_platt_logits)
    ]

    all_p = all_logits.softmax(dim=-1)
    all_y_hat = all_p.argmax(dim=-1)
    acc = (all_y == all_y_hat).float().mean()
    logits_ece, _ = calibration(
        all_y, all_y_hat, all_p[torch.arange(all_p.size(0)), all_y_hat]
    )

    all_cal_logits = all_logits - all_platt_logits
    all_cal_p = all_cal_logits.softmax(dim=-1)
    all_cal_y_hat = all_cal_p.argmax(dim=-1)
    cal_acc = (all_y == all_cal_y_hat).float().mean()
    cal_ece, _ = calibration(
        all_y, all_y_hat, all_cal_p[torch.arange(all_cal_p.size(0)), all_cal_y_hat]
    )

    return_dict = {
        "N": all_y.size(0),
        # OE Change: acc -> first_token_acc
        "first_token_acc": acc.item(),
        "logits_ece": logits_ece,
        # OE Change: acc -> first token acc
        "cal_first_token_acc": cal_acc.item(),
        "cal_ece": cal_ece,
    }

    # CHANGE FOR OE
    # generate the entire remainder of the sequence for each element

    # all_oe_inputs = []
    all_acc = {c: [] for c in comparison_strategies}
    all_oe_target_strings, all_output_strings, all_question_strings = [], [], []

    all_cal_p_index = 0

    for inputs in tqdm(loader, leave=False):

        inputs = prepare_batch(tokenizer, inputs, prompt_style=prompt_style)
        inputs = collate_fn(inputs)

        # extract first token
        first_token = all_cal_p[
            all_cal_p_index : all_cal_p_index + len(inputs["input_ids"])
        ]
        all_cal_p_index += len(inputs["input_ids"])

        target_start_idx, oe_inputs_base, oe_targets = extract_oe_inputs(
            tokenizer, inputs
        )

        oe_inputs = collate_fn(oe_inputs_base)
        oe_inputs = {k: v.to(device) for k, v in oe_inputs.items()}

        oe_inputs_extended = collate_fn(oe_inputs_base)
        oe_inputs_extended = {k: v.to(device) for k, v in oe_inputs_extended.items()}

        # add the calibrated first token
        oe_inputs_extended["input_ids"] = torch.cat(
            [
                oe_inputs_extended["input_ids"],
                first_token.argmax(dim=-1).unsqueeze(dim=1).to(device),
            ],
            dim=-1,
        )
        oe_inputs_extended["attention_mask"] = torch.cat(
            [
                oe_inputs_extended["attention_mask"],
                torch.ones(oe_inputs_extended["attention_mask"].shape[0], 1).to(device),
            ],
            dim=-1,
        )

        # import pdb; pdb.set_trace()

        if isinstance(model, PeftModel):
            model.set_adapter("default")

        # generate 30 - 1 more tokens
        outputs = model.generate(
            **oe_inputs_extended, max_new_tokens=max_new_tokens - 1
        )

        # convert those new tokens to the generated strings
        output_strings = tokenizer.batch_decode(
            outputs[..., target_start_idx:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # these are the ground truth strings for this batch
        oe_target_strings = tokenizer.batch_decode(
            oe_targets, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        question_strings = tokenizer.batch_decode(
            oe_inputs["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # prepare the calibration query with open ended text
        # the calculation of the accuracy is done within this function
        for c in comparison_strategies:
            _, _, acc = prepare_oe_calibration_query(
                tokenizer,
                oe_target_strings,
                output_strings,
                question_strings,
                format=query_format,
                comparison_strategy=c,
            )

            acc = acc.to(device)

            [
                l.append(v)
                for l, v in zip(
                    (all_acc[c]),
                    accelerator.gather_for_metrics((acc)),
                )
            ]

        [
            l.append(v)
            for l, v in zip(
                (all_oe_target_strings, all_output_strings, all_question_strings),
                accelerator.gather_for_metrics(
                    (oe_target_strings, output_strings, question_strings)
                ),
            )
        ]

    for c in comparison_strategies:
        all_acc_c = np.array([e.cpu().numpy() for e in all_acc[c]]).flatten()
        return_dict[f"{c}_acc"] = np.sum(all_acc_c) / len(all_acc_c)

        c_ece, _ = calibration(
            np.ones_like(all_acc_c),
            all_acc_c,
            all_cal_p[torch.arange(all_cal_p.size(0)), all_cal_y_hat].cpu().numpy(),
        )

        return_dict[f"{c}_ece"] = c_ece

    return return_dict


VERBAL_ELICITATION_UNC_QUERIES = {
    "ve_2c": "".join([
        "Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\n",
        "For example:\n\n",
        "Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n",
    ]),
    "ve_21": "".join([
        "Provide the probability that your guess is correct. Give ONLY the probability, no other words or explanation.\n\n",
        "For example:\n\n",
        "Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n",
    ]),
    "ve_22": "".join([
        "Provide the probability that each of your guesses is correct. Give ONLY the probabilities, no other words or explanation.\n\n",
        "For example:\n\n",
        "P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n",
        "P2: <the probability between 0.0 and 1.0 that G2 is correct, without any extra commentary whatsoever; just the probability!>\n",
    ]),
    "ve_24": "".join([
        "Provide the probability that each of your guesses is correct. Give ONLY the probabilities, no other words or explanation.\n\n",
        "For example:\n\n",
        "P1: <the probability between 0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just the probability!>\n",
        "P2: <the probability between 0.0 and 1.0 that G2 is correct, without any extra commentary whatsoever; just the probability!>\n",
        "P3: <the probability between 0.0 and 1.0 that G3 is correct, without any extra commentary whatsoever; just the probability!>\n",
        "P4: <the probability between 0.0 and 1.0 that G4 is correct, without any extra commentary whatsoever; just the probability!>\n",
    ]),
}

def parse_verbal_elicitation_oe(
    output_string,
):
    parts = output_string.split("\n")[:2]

    if len(parts) == 2:
        output, prob = parts 
    else:
        output = parts[0]
        if "Probability" in output:
            prob = output
            output = ""
        else:
            prob = ""

    try:
        prob = float(prob.split(":")[-1].strip())
    except:
        prob = 0.5

    y = torch.tensor([prob > 0.5])
    logits = torch.tensor([[1 - prob, prob]])

    return output, y, logits

@torch.inference_mode()
def evaluate_verbal_elicitation_oe(
    accelerator,
    model,
    tokenizer,
    loader,
    prompt_style="oe",
    comparison_strategies=None,
    max_new_tokens=30,
    output_row_path=None,
    **_,
):
    assert prompt_style == "oe"
    assert (not comparison_strategies is None) and len(comparison_strategies) > 0

    device = accelerator.device

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )

    all_unc_y, all_unc_logits = {c: [] for c in comparison_strategies}, {
        c: [] for c in comparison_strategies
    }
    all_acc = {c: [] for c in comparison_strategies}
    all_oe_target_strings, all_output_strings, all_question_strings = [], [], []

    output_generator = generate_output(
        accelerator,
        model,
        tokenizer,
        loader,
        prompt_style=prompt_style,
        generation_config=generation_config,
    )

    # i = 0
    # for example in output_generator:
    #     from pprint import pprint
    #     pprint(example)

    #     parse_verbal_elicitation_oe(example["output"])
    #     # print("\n")
    #     i += 1
    #     if i > 20:
    #         break
    
    # print(1/0)

    for example in output_generator:
        output, raw_input, target = (
            example["output"],
            example["raw_input"],
            example["target"],
        )

        output, unc_y, unc_logits = parse_verbal_elicitation_oe(output)

        unc_y = unc_y.to(device)
        unc_logits = unc_logits.to(device)

        input_question_string = example["context"]
        oe_target_strings = [target]
        output_strings = [output]
        question_strings = [input_question_string]

        for c in comparison_strategies:
            acc = torch.tensor(
                grade_oe_preds(
                    oe_target_strings,
                    output_strings,
                    question_strings,
                    comparison_strategy=c,
                )
            ).to(device)

            [
                l.append(v)
                for l, v in zip(
                    (all_unc_y[c], all_unc_logits[c], all_acc[c]),
                    (unc_y, unc_logits, acc),
                )
            ]

        [
            l.append(v)
            for l, v in zip(
                (all_oe_target_strings, all_output_strings, all_question_strings),
                (oe_target_strings, output_strings, question_strings),
            )
        ]

    all_oe_target_strings, all_output_strings, all_question_strings = (
        sum(all_oe_target_strings, []),
        sum(all_output_strings, []),
        sum(all_question_strings, []),
    )

    return_dict = {}
    # note if using multiple gpus/processes, these string lists will be incomplete.
    dump = {
        "oe_target_strings": all_oe_target_strings,
        "output_strings": all_output_strings,
        "question_strings": all_question_strings,
    }

    for c in comparison_strategies:

        all_unc_y_c, all_unc_p, all_acc_c = [
            torch.cat(l, dim=0) for l in (all_unc_y[c], all_unc_logits[c], all_acc[c])
        ]

        acc = all_acc_c.float().mean()

        all_unc_y_c = (
            all_unc_y_c.unsqueeze(-1) == torch.arange(2).to(device)
        ).long().argmax(dim=-1)
        all_unc_y_hat = all_unc_p.argmax(dim=-1)
        unc_acc = (all_unc_y_c == all_unc_y_hat).float().mean()
        unc_ece, _ = calibration(
            all_unc_y_c,
            all_unc_y_hat,
            all_unc_p[torch.arange(all_unc_p.size(0)), all_unc_y_hat],
        )

        return_dict.update(
            {
                f"{c}_acc": acc.item(),
                f"{c}_unc_acc": unc_acc.item(),
                f"{c}_unc_ece": unc_ece,
                "N": all_unc_p.size(0),
            }
        )

        dump.update(
            {
                f"{c}_acc": all_acc_c.cpu().numpy(),
                f"{c}_all_unc_y": all_unc_y_c.cpu().numpy(),
                f"{c}_all_unc_y_hat": all_unc_y_hat.cpu().numpy(),
                f"{c}_unc_acc": unc_acc.item(),
                f"{c}_unc_ece": unc_ece,
            }
        )

    if accelerator.num_processes == 1 and output_row_path is not None:
        #create parent dir if it doesn't exist
        import os
        os.makedirs(os.path.dirname(output_row_path), exist_ok=True)
        pd.DataFrame(dump).to_csv(output_row_path, escapechar="\\")

    return return_dict