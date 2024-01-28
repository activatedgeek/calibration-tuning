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
    collate_fn = StringDataCollator(tokenizer)

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
                    output_scores=True
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

        yield from outputs
