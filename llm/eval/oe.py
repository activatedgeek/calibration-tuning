from tqdm.auto import tqdm
import torch
from peft import PeftModel
import pandas as pd
import numpy as np

from ..datasets.llm_utils import (
    DataCollatorForSupervisedDataset,
    extract_qa_exact,
    prepare_batch
)
from ..datasets.llm_utils_oe import (
    extract_qa_oe,
    extract_oe_inputs,
    prepare_oe_calibration_query
)
from .third_party.calibration import calibration


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
    assert(prompt_style=="oe")
    assert((not comparison_strategies is None) and len(comparison_strategies) > 0)
    device = accelerator.device
    collate_fn = DataCollatorForSupervisedDataset(tokenizer)

    query_token_vec = None
    # all_oe_inputs = []
    all_unc_y, all_unc_logits = {c : [] for c in comparison_strategies}, {c : [] for c in comparison_strategies}
    all_acc = {c : [] for c in comparison_strategies}
    all_oe_target_strings, all_output_strings, all_question_strings = [], [], []

    for inputs in tqdm(loader, leave=False):
        inputs = prepare_batch(tokenizer, inputs, prompt_style=prompt_style)
        inputs = collate_fn(inputs)

        # get the target separation between prompt and answer
        target_start_idx, oe_inputs, oe_targets = extract_oe_inputs(tokenizer, inputs)
        oe_inputs = collate_fn(oe_inputs)
        oe_inputs = {k: v.to(device) for k, v in oe_inputs.items()}

        # these are the ground truth strings for this batch
        oe_target_strings = tokenizer.batch_decode(oe_targets, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        

        if isinstance(model, PeftModel):
            model.set_adapter("default")

        # generate 30 more tokens
        outputs = model.generate(**oe_inputs, max_new_tokens=max_new_tokens)

        # convert those new tokens to the generated strings 
        output_strings = tokenizer.batch_decode(outputs[...,target_start_idx:], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        question_strings = tokenizer.batch_decode(oe_inputs['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # prepare the calibration query with open ended text
        # the calculation of the accuracy is done within this function
        for c in comparison_strategies:
            query_inputs, query_token_vec, acc = prepare_oe_calibration_query(
                tokenizer, oe_target_strings, output_strings, question_strings, format=query_format, comparison_strategy=c
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
                accelerator.gather_for_metrics((oe_target_strings, output_strings, question_strings)),
            )
        ]          

    return_dict = {
        "N": len(all_oe_target_strings),
    }

    all_oe_target_strings, all_output_strings, all_question_strings = sum(all_oe_target_strings, []), sum(all_output_strings, []), sum(all_question_strings, [])
    dump = {
        "oe_target_strings": all_oe_target_strings, 
        "output_strings": all_output_strings,
        "question_strings": all_question_strings,
    }

    for c in comparison_strategies:

        query_token_vec = query_token_vec.to(device)
        all_unc_y_c, all_unc_logits_c, all_acc_c = [torch.cat(l, dim=0) for l in (all_unc_y[c], all_unc_logits[c], all_acc[c])]

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
            }
        )

        dump.update(
            {
                f"{c}_acc": all_acc_c.cpu().numpy(),
                f"{c}_all_unc_y": all_unc_y_c.cpu().numpy(),
                f"{c}_all_unc_y_hat": all_unc_y_hat.cpu().numpy(),
            }
        )

    if output_row_path is not None:
        pd.DataFrame(dump).to_csv(output_row_path, escapechar='\\')


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
    top_p = 0.95,
    k = 5
):
    assert(prompt_style=="oe")
    assert((not comparison_strategies is None) and len(comparison_strategies) > 0)
    device = accelerator.device
    collate_fn = DataCollatorForSupervisedDataset(tokenizer)

    query_token_vec = None
    # all_oe_inputs = []
    all_prob = {c : [] for c in comparison_strategies}
    all_prob_vectors = []
    all_acc = {c : [] for c in comparison_strategies}
    all_oe_target_strings, all_output_strings, all_question_strings = [], [], []

    for inputs in tqdm(loader, leave=False):
        inputs = prepare_batch(tokenizer, inputs, prompt_style=prompt_style)
        inputs = collate_fn(inputs)

        # get the target separation between prompt and answer
        target_start_idx, oe_inputs, oe_targets = extract_oe_inputs(tokenizer, inputs)
        oe_inputs = collate_fn(oe_inputs)
        oe_inputs = {k: v.to(device) for k, v in oe_inputs.items()}

        # these are the ground truth strings for this batch
        oe_target_strings = tokenizer.batch_decode(oe_targets, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        

        if isinstance(model, PeftModel):
            model.set_adapter("default")

        # generate 30 more tokens

        outputs_list = []
        for i in range(k):
            
            outputs = model.generate(
                **oe_inputs, max_new_tokens=max_new_tokens,
                top_p = top_p,
                do_sample = True
            )

            assert(len(outputs) == 1)

            outputs_list.append(outputs[0,target_start_idx:])

        outputs_list_length_normalized = np.array([
            np.exp(np.sum(np.log(o.detach().cpu().numpy())) / len(o)) for o in outputs_list
        ])

        argmax_i = np.argmax(outputs_list_length_normalized)

        prob = outputs_list_length_normalized[argmax_i] / np.sum(outputs_list_length_normalized)

        outputs_argmax = outputs_list[argmax_i].unsqueeze(dim=0)

        # outputs = model.generate(**oe_inputs, max_new_tokens=max_new_tokens)

        # convert those new tokens to the generated strings 
        output_strings = tokenizer.batch_decode(outputs_argmax, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        question_strings = tokenizer.batch_decode(oe_inputs['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # prepare the calibration query with open ended text
        # the calculation of the accuracy is done within this function
        for c in comparison_strategies:
            _, _, acc = prepare_oe_calibration_query(
                tokenizer, oe_target_strings, output_strings, question_strings, format=query_format, comparison_strategy=c
            )

            acc = acc.to(device)

            [
                l.append(v)
                for l, v in zip(
                    (all_prob[c], all_acc[c]),
                    accelerator.gather_for_metrics((prob, acc)),
                )
            ]          

        [
            l.append(v)
            for l, v in zip(
                (all_oe_target_strings, all_output_strings, all_question_strings, all_prob_vectors),
                accelerator.gather_for_metrics((oe_target_strings, output_strings, question_strings, outputs_list_length_normalized)),
            )
        ]          

    return_dict = {
        "N": len(all_oe_target_strings),
    }

    all_oe_target_strings, all_output_strings, all_question_strings = sum(all_oe_target_strings, []), sum(all_output_strings, []), sum(all_question_strings, [])
    dump = {
        "oe_target_strings": all_oe_target_strings, 
        "output_strings": all_output_strings,
        "question_strings": all_question_strings,
        "all_prob_vectors": all_prob_vectors
    }

    for c in comparison_strategies:

        all_acc_c = np.concatenate([c.cpu() for c in all_acc[c]], axis=0)

        acc = all_acc_c.mean()
        # import pdb; pdb.set_trace()
        assert(len(all_prob[c]) == len(all_acc_c))
        ece, _ = calibration(
            np.ones_like(all_prob[c]),
            all_acc_c.astype(dtype=np.dtype('i4')),
            np.array(all_prob[c]),
        )


        return_dict.update(
            {
                f"{c}_acc": acc.item(),
                f"{c}_ece": ece,
            }
        )

        dump.update(
            {
                f"{c}_acc": all_acc_c,
                # f"{c}_all_unc_y": all_unc_y_c.cpu().numpy(),
                # f"{c}_all_unc_y_hat": all_unc_y_hat.cpu().numpy(),
            }
        )

    if output_row_path is not None:
        pd.DataFrame(dump).to_csv(output_row_path, escapechar='\\')


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
            calib_inputs = prepare_batch(tokenizer, calib_inputs, prompt_style=prompt_style)
            #calib_inputs = prepare_batch(tokenizer, calib_inputs)
            calib_inputs = collate_fn(calib_inputs)

            calib_inputs = {k: v.to(device) for k, v in calib_inputs.items()}
            calib_outputs = model(**calib_inputs)

            # OE changes: extract oe inputs

            calib_target_start_idx, _, _ = extract_oe_inputs(tokenizer, calib_inputs)
            _c_logits = torch.squeeze(calib_outputs.logits[:,calib_target_start_idx,:], dim=1) # first token only

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
        logits = torch.squeeze(outputs.logits[:,target_start_idx,:], dim=1)
        y = oe_targets[:,0]
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
    all_acc = {c : [] for c in comparison_strategies}
    all_oe_target_strings, all_output_strings, all_question_strings = [], [], []

    all_cal_p_index = 0

    for inputs in tqdm(loader, leave=False):

        inputs = prepare_batch(tokenizer, inputs, prompt_style=prompt_style)
        inputs = collate_fn(inputs)

        # extract first token
        first_token = all_cal_p[all_cal_p_index : all_cal_p_index + len(inputs['input_ids'])]
        all_cal_p_index += len(inputs['input_ids'])

        target_start_idx, oe_inputs_base, oe_targets = extract_oe_inputs(tokenizer, inputs)

        oe_inputs = collate_fn(oe_inputs_base)
        oe_inputs = {k: v.to(device) for k, v in oe_inputs.items()}

        oe_inputs_extended = collate_fn(oe_inputs_base)
        oe_inputs_extended = {k: v.to(device) for k, v in oe_inputs_extended.items()}

        # add the calibrated first token
        oe_inputs_extended['input_ids'] = torch.cat(
            [
                oe_inputs_extended['input_ids'], 
                first_token.argmax(dim=-1).unsqueeze(dim=1).to(device)
            ],
            dim=-1
        )
        oe_inputs_extended['attention_mask'] = torch.cat(
            [
                oe_inputs_extended['attention_mask'], 
                torch.ones(oe_inputs_extended['attention_mask'].shape[0], 1).to(device)
            ],
            dim=-1
        )

        # import pdb; pdb.set_trace()

        if isinstance(model, PeftModel):
            model.set_adapter("default")

        # generate 30 - 1 more tokens
        outputs = model.generate(**oe_inputs_extended, max_new_tokens=max_new_tokens - 1)

        # convert those new tokens to the generated strings 
        output_strings = tokenizer.batch_decode(outputs[...,target_start_idx:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        

        # these are the ground truth strings for this batch
        oe_target_strings = tokenizer.batch_decode(oe_targets, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        question_strings = tokenizer.batch_decode(oe_inputs['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # prepare the calibration query with open ended text
        # the calculation of the accuracy is done within this function
        for c in comparison_strategies:
            _, _, acc = prepare_oe_calibration_query(
                tokenizer, oe_target_strings, output_strings, question_strings, format=query_format, comparison_strategy=c
            )

            acc = acc.to(device)

            all_acc[c].append(accelerator.gather_for_metrics(acc))   

        [
            l.append(v)
            for l, v in zip(
                (all_oe_target_strings, all_output_strings, all_question_strings),
                accelerator.gather_for_metrics((oe_target_strings, output_strings, question_strings)),
            )
        ]          

    for c in comparison_strategies:
        all_acc_c = np.array([e.cpu().numpy() for e in all_acc[c]]).flatten()
        return_dict[f"{c}_acc"] = np.sum(all_acc_c)/len(all_acc_c)

        c_ece, _ = calibration(
            np.ones_like(all_acc_c),
            all_acc_c,
            all_cal_p[torch.arange(all_cal_p.size(0)), all_cal_y_hat].cpu().numpy()
        )

        return_dict[f"{c}_ece"] = c_ece

    return return_dict
    