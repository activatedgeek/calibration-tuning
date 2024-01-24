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
    accelerator, model, tokenizer, loader, prompt_style="oe", generation_config=None, outputs_only=True
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

        ## NOTE: Verify output extraction pre-condition.
        assert (
            generation_inputs.get("input_ids")
            == generation_outputs[:, : generation_inputs.get("input_ids").size(-1)]
        ).all()

        examples = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
        examples = [
            {
                **o,
                "output": tokenizer.decode(
                    t[inp.size(0) :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
            } if outputs_only else {
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
                # "raw_output": tokenizer.decode(
                #     t,
                #     skip_special_tokens=True,
                #     clean_up_tokenization_spaces=False,
                # ),
            }
            for o, inp, t in zip(
                examples, generation_inputs.get("input_ids"), generation_outputs
            )
        ]

        # for k in outputs:
        #     print(k["context"])
        #     print(k["target_prompt"])
        #     print("\n##################\n##### OUTPUT #####\n##################\n")
        #     print(k["output"])
        #     print("\n*****************************************************************\n*****************************************************************\n")

        yield from examples