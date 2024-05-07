import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from peft import PeftModel

from llm.datasets import LabeledStringDataCollator


def wrapped_generate_output(model, tokenizer, generation_inputs, generation_config):
    while True:
        try:
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            generation_outputs = model.generate(
                **generation_inputs, 
                eos_token_id=terminators,
                generation_config=generation_config
            )
            return generation_outputs
        except Exception as e:
            generation_outputs = []
            new_bs = max(1, generation_inputs["input_ids"].size(0) // 2)
            for i in range(0, generation_inputs["input_ids"].size(0), new_bs):
                inputs = {k: v[i : i + new_bs] for k, v in generation_inputs.items()}
                _outputs = wrapped_generate_output(model, inputs, generation_config)
                generation_outputs.append(_outputs)
            return torch.cat(generation_outputs, dim=0)

def generate_output(
    accelerator,
    model,
    tokenizer,
    loader,
    generation_config=None,
    generation_config_sampling=None,
    n_samples=0,
    log_dir=None,
):
    collate_fn = LabeledStringDataCollator(tokenizer)

    all_outputs = []

    for inputs in tqdm(loader):
        inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]
        targets = [inp.pop("target") for inp in inputs]

        generation_inputs = {
            k: v.to(accelerator.device) for k, v in collate_fn(inputs).items()
        }

        if isinstance(model, PeftModel):
            model.set_adapter("default")
        
        generation_outputs = wrapped_generate_output(
            model, tokenizer, generation_inputs, generation_config
        )

        generations = tokenizer.batch_decode(
            generation_outputs[:, generation_inputs.get("input_ids").size(-1) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        # for x in generations:
        #     print(x)

        outputs = [
            {**inp, "target": tgt, "output": gen}
            for inp, tgt, gen in zip(inputs, targets, generations)
        ]

        if n_samples:
            assert generation_config_sampling is not None

            # https://github.com/huggingface/transformers/issues/14498#issuecomment-977909651
            sampled_outputs = [
                model.generate(
                    **generation_inputs,
                    generation_config=generation_config_sampling,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                for _ in range(n_samples)
            ]

            outputs = [
                {
                    **o,
                    "sampled_outputs": tokenizer.batch_decode(
                        so["sequences"][
                            :, generation_inputs.get("input_ids").size(-1) :
                        ]
                    ),
                    "sampled_log_probs": F.log_softmax(
                        torch.cat(so["scores"], dim=0), dim=-1
                    ),
                }
                for o, so in zip(outputs, sampled_outputs)
            ]

        all_outputs.extend(outputs)

    if log_dir is not None:
        df = pd.DataFrame(all_outputs)
        ## NOTE: Avoid spec errors when loading for labeling.
        df["query_label"] = -1

        df.to_csv(f"{log_dir}/rows_{accelerator.process_index}.csv", index=False)
