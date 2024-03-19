from functools import partial
import fire
import torch
from transformers import GenerationConfig

from llm.datasets import LabeledStringDataCollator, prepare_uncertainty_query
from llm.models import get_model
from llm.models.peft import get_lora_model


@torch.inference_mode
def generate_answer(query, model=None, tokenizer=None, max_new_tokens=None):
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    collate_fn = LabeledStringDataCollator(tokenizer)

    str_inputs = [{"context": query}]
    inputs = {k: v.cuda() for k, v in collate_fn(str_inputs).items()}

    with model.disable_adapter():
        outputs = model.generate(**inputs, generation_config=generation_config)

    str_outputs = tokenizer.batch_decode(
        outputs[:, inputs.get("input_ids").size(-1) :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    q_str_inputs, _, q_token_vec = prepare_uncertainty_query(
        tokenizer,
        str_inputs,
        [""] * len(str_inputs),
        str_outputs,
        strategy="substring",
        format="roman_choice",
    )
    q_inputs = inputs = {k: v.cuda() for k, v in collate_fn(q_str_inputs).items()}

    q_outputs = model(**q_inputs)
    q_logits = q_outputs.logits[..., -1, q_token_vec].softmax(dim=-1)

    response = str_outputs[0].strip()
    p_correct = q_logits[:, 1].item()

    return response, p_correct


def main(max_new_tokens=100):
    tokenizer, model = get_model("llama2_13b_chat", device_map="auto")

    model = get_lora_model(
        model,
        peft_id_or_dir="calibration-tuning/Llama-2-13b-chat-ct-oe",
        adapter_name="query",
    )

    model.eval()

    respond_fn = partial(
        generate_answer, model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens
    )

    while True:
        query = input("(Enter query)> ")

        response, p_correct = respond_fn(query)

        print(f"(Pinocchio says with {p_correct * 100:.1f}% confidence)> {response}")
        print()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
