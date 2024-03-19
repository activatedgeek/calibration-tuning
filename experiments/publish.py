import os

from llm.models import get_model
from llm.models.peft import get_lora_model


def main(
    model_name=None,
    query_peft_dir=None,
    hf_username=None,
    hf_repo_id=None,
    hf_token=None,
):
    hf_username = hf_username or os.environ.get("HUGGING_FACE_USERNAME")
    hf_token = hf_token or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if hf_username is None or hf_token is None:
        raise ValueError(
            f"Missing hf_username (env: HUGGING_FACE_USERNAME) and/or hf_token (env: HUGGING_FACE_HUB_TOKEN)"
        )

    model_id = f"{hf_username}/{hf_repo_id}"

    _, model = get_model(
        model_name,
        device_map="cpu",
    )

    model = get_lora_model(
        model,
        peft_dir=query_peft_dir,
        is_trainable=False,
        adapter_name="default",
    )

    print(f'Pushing model "{model_id}" to HuggingFace Hub.')

    model.push_to_hub(model_id, private=True, token=hf_token)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
