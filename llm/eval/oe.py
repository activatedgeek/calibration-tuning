from tqdm.auto import tqdm
import torch
from peft import PeftModel
import pandas as pd

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
    input_generation_path=None,
):
    assert(prompt_style=="oe")
    assert((not comparison_strategies is None) and len(comparison_strategies) > 0)
    device = accelerator.device
    collate_fn = DataCollatorForSupervisedDataset(tokenizer)

    query_token_vec = {}
    all_oe_inputs = []
    all_unc_y, all_unc_logits = {c : [] for c in comparison_strategies}, {c : [] for c in comparison_strategies}
    all_acc = {c : [] for c in comparison_strategies}
    all_oe_target_strings, all_output_strings, all_question_strings = [], [], []

    if input_generation_path is not None:
        df_load = pd.read_csv(input_generation_path, escapechar="\\")
        loaded_output_strings = [ [s] for s in df_load["output_strings"]]
        loaded_question_strings = [ [s] for s in df_load["question_strings"]]
        prior_comparison_results = []
        for _, row in df_load.iterrows():
            this_entry = {}
            for c in comparison_strategies:
                prior_acc = row[f"{c}_acc"]
                this_entry[c] = prior_acc
            prior_comparison_results.append([this_entry])
    else:
        loaded_output_strings = [None for _ in inputs]
        loaded_question_strings = [None for _ in inputs]
        prior_comparison_results = [None for _ in inputs]

    for inputs, previous_output, previous_questions, prior_comparison_result_batch in tqdm(zip(loader, loaded_output_strings, loaded_question_strings, prior_comparison_results), leave=False):

        assert(len(inputs["context"]) == 1)

        question_strings = inputs["context"]
        if input_generation_path is not None:
            assert(previous_questions[0] == question_strings[0])

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
        if input_generation_path is not None:
            output_strings = previous_output
        else:
            outputs = model.generate(**oe_inputs, max_new_tokens=max_new_tokens)

            # convert those new tokens to the generated strings 
            output_strings = tokenizer.batch_decode(outputs[...,target_start_idx:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        if prior_comparison_result_batch is not None:
            prior_comparison_result_dict = {
                c : [entry[c] for entry in prior_comparison_result_batch]
                for c in comparison_strategies
            } 

        # prepare the calibration query with open ended text
        # the calculation of the accuracy is done within this function
        for c in comparison_strategies:

            if prior_comparison_result_batch is not None:
                prior_comparison_result_c = prior_comparison_result_dict[c]
            else:
                prior_comparison_result_c = None

            query_inputs, query_token_vec[c], acc = prepare_oe_calibration_query(
                tokenizer, oe_target_strings, output_strings, question_strings, format=query_format, comparison_strategy=c, prior_comparison_results=prior_comparison_result_c
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

        query_token_vec[c] = query_token_vec[c].to(device)
        all_unc_y_c, all_unc_logits_c, all_acc_c = [torch.cat(l, dim=0) for l in (all_unc_y[c], all_unc_logits[c], all_acc[c])]

        acc = all_acc_c.float().mean()

        all_unc_y_c, all_unc_p = (
            (all_unc_y_c.unsqueeze(-1) == query_token_vec[c]).long().argmax(dim=-1),
            all_unc_logits_c[:, query_token_vec[c]].softmax(dim=-1),
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
                f"{c}_all_unc_p": all_unc_p[torch.arange(all_unc_p.size(0)), all_unc_y_hat].detach().cpu().numpy(),
            }
        )

    if output_row_path is not None:
        pd.DataFrame(dump).to_csv(output_row_path, escapechar="\\")


    return return_dict
