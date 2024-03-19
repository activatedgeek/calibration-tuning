import os

from llm.logging import entrypoint
from llm.models import get_model
from llm.models.peft import get_lora_model


def main(
    model_name=None,
    query_peft_dir=None,
    hf_model_id=None,
    hf_token=None,
):
    hf_token = hf_token or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token is None:
        raise ValueError(
            f"Missing hf_username (env: HUGGING_FACE_USERNAME) and/or hf_token (env: HUGGING_FACE_HUB_TOKEN)"
        )

    _, model = get_model(
        model_name,
        device_map="cpu",
    )

    model = get_lora_model(
        model,
        peft_id_or_dir=query_peft_dir,
        is_trainable=False,
        adapter_name="default",
    )

    print(f'Pushing model "{hf_model_id}" to HuggingFace Hub.')

    model.push_to_hub(hf_model_id, private=True, token=hf_token)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
