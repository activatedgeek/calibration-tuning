import torch
from tqdm.auto import tqdm
from peft import PeftModel

from llm.datasets.llm_utils import (
    prepare_batch,
    DataCollatorForSupervisedDataset,
)


def generate_output(
    accelerator,
    model,
    tokenizer,
    loader,
    prompt_style="oe",
    generation_config=None,
    generation_config_sampling=None,
    n_samples=0,
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

        if n_samples:
            assert generation_config_sampling is not None

            # https://github.com/huggingface/transformers/issues/14498#issuecomment-977909651
            sampled_outputs = [
                model.generate(
                    **generation_inputs,
                    generation_config=generation_config_sampling,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                for _ in range(n_samples)
            ]

        ## NOTE: Verify output extraction pre-condition.
        assert (
            generation_inputs.get("input_ids")
            == generation_outputs[:, : generation_inputs.get("input_ids").size(-1)]
        ).all()

        examples_pre = [
            dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())
        ]
        examples = []
        for o, inp, t in zip(
            examples_pre, generation_inputs.get("input_ids"), generation_outputs
        ):
            example = {
                **o,
                "output": tokenizer.decode(
                    t[inp.size(0) :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
                # "raw_input": tokenizer.decode(
                #     inp,
                #     skip_special_tokens=True,
                #     clean_up_tokenization_spaces=False,
                # ),
                "target": o["target"],
            }

            if n_samples:
                example["sampled_outputs"] = [
                    tokenizer.decode(
                        entry["sequences"][i][inp.size(0) :],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for entry in sampled_outputs
                ]
                example["sampled_log_probs"] = [
                    torch.nn.functional.log_softmax(
                        torch.cat(entry["scores"], dim=0), dim=-1
                    )
                    for entry in sampled_outputs
                ]
            examples.append(example)

        yield from examples
