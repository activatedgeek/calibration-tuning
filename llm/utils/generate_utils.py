# import os
# import wandb
# import pandas as pd
# import torch
# from accelerate import Accelerator
from tqdm.auto import tqdm
from peft import PeftModel
# from transformers import GenerationConfig

# from llm.datasets import get_dataset, get_loader
from llm.datasets.llm_utils import (
    LMText,
    prepare_batch,
    DataCollatorForSupervisedDataset,
)
# from llm.models import get_model
# from llm.models.peft import get_lora_model
# from llm.logging import entrypoint

# from llm.datasets.llm_utils_oe import prepare_oe_calibration_query


def generate_output(
    accelerator, 
    model, 
    tokenizer, 
    loader, 
    prompt_style="oe", 
    generation_config=None, 
    generation_config_sampling=None,
    k=None
):
    if isinstance(model, PeftModel):
        model.set_adapter("default")

    collate_fn = DataCollatorForSupervisedDataset(tokenizer)

    for inputs in tqdm(loader):
        generation_inputs = prepare_batch(
            tokenizer,
            ## Skip "target" for generation.
            {k: v for k, v in inputs.items() if k != "target"},
            prompt_style=prompt_style,
        )
        generation_inputs = {
            k: v.to(accelerator.device)
            for k, v in collate_fn(generation_inputs).items()
        }

        generation_outputs = model.generate(
            **generation_inputs, generation_config=generation_config
        )

        # if k is not None:
        #     assert(generation_config_sampling is not None)
        #     assert(k > 0)
        #     #https://github.com/huggingface/transformers/issues/14498#issuecomment-977909651
        #     sampled_outputs = [
        #         model.generate(
        #             **generation_inputs,
        #             generation_config=generation_config_sampling,
        #             output_scores=True
        #         )
        #         for _ in range(k)
        #     ]
        # else:
        sampled_outputs = [[] for _ in range(len(generation_outputs))]

        ## NOTE: Verify output extraction pre-condition.
        assert (
            generation_inputs.get("input_ids")
            == generation_outputs[:, : generation_inputs.get("input_ids").size(-1)]
        ).all()

        examples_pre = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
        examples = []
        for o, inp, t, s in zip(examples_pre, generation_inputs.get("input_ids"), generation_outputs, sampled_outputs):
            example = {
                **o,
                "output": tokenizer.decode(
                    t[inp.size(0) :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
                "raw_input": tokenizer.decode(
                    inp,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
                "target": o["target"],
            }
            if k is not None:
                assert(False)
                # example["sampled_outputs"] = [
                #     tokenizer.decode(
                #         entry[inp.size(0) :],
                #         skip_special_tokens=True,
                #         clean_up_tokenization_spaces=False,
                #     )
                #     for entry in s
                # ]
                # example["sampled_probs"] = [
                #     entry[inp.size(0) :]
                #     for entry in s
                # ]

            examples.append(example)

        yield from examples