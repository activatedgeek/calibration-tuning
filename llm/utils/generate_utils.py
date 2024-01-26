import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from peft import PeftModel

from llm.datasets.llm_utils import StringDataCollator


def generate_output(
    accelerator,
    model,
    tokenizer,
    loader,
    generation_config=None,
    generation_config_sampling=None,
    n_samples=0,
):
    if isinstance(model, PeftModel):
        model.set_adapter("default")

    collate_fn = StringDataCollator(tokenizer)

    for inputs in tqdm(loader):
        ## Convert to list of dictionaries, without target for generation.
        targets = inputs["target"]
        inputs = {k: v for k, v in inputs.items() if k != "target"}
        inputs = [dict(zip(inputs.keys(), vals)) for vals in zip(*inputs.values())]

        generation_inputs = {
            k: v.to(accelerator.device) for k, v in collate_fn(inputs).items()
        }

        generation_outputs = model.generate(
            **generation_inputs, generation_config=generation_config
        )

        ## Verify output extraction pre-condition.
        # assert (
        #     generation_inputs.get("input_ids")
        #     == generation_outputs[:, : generation_inputs.get("input_ids").size(-1)]
        # ).all()
        padded_input_len = generation_inputs.get("input_ids").size(-1)

        outputs = [
            {
                **inp,
                "target": tgt,
                "output": tokenizer.decode(
                    go[padded_input_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
            }
            for inp, tgt, go in zip(inputs, targets, generation_outputs)
        ]

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

            outputs = [
                {
                    **o,
                    "sampled_outputs": tokenizer.batch_decode(
                        so["sequences"][:, padded_input_len:]
                    ),
                    "sampled_log_probs": F.log_softmax(
                        torch.cat(so["scores"], dim=0), dim=-1
                    ),
                }
                for o, so in zip(outputs, sampled_outputs)
            ]

        yield from outputs
