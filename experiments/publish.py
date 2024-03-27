import os
from datasets import DatasetDict

from llm.datasets import get_dataset
from llm.models import get_model
from llm.models.peft import get_lora_model


def main(
    dataset=None,
    model_name=None,
    query_peft_dir=None,
    hf_hub_id=None,
    hf_token=None,
):
    hf_token = hf_token or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token is None:
        raise ValueError(
            f"Missing hf_username (env: HUGGING_FACE_USERNAME) and/or hf_token (env: HUGGING_FACE_HUB_TOKEN)"
        )

    if model_name is not None:
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

        print(f'Pushing model "{hf_hub_id}" to HuggingFace Hub.')

        model.push_to_hub(hf_hub_id, private=True, token=hf_token)
    elif dataset is not None:
        train_data, val_data, _ = get_dataset(
            dataset,
            num_workers=8,
            use_cache=True,
        )

        dataset = DatasetDict({"train": train_data, "validation": val_data})

        print(f'Pushing dataset "{hf_hub_id}" to HuggingFace Hub.')

        dataset.push_to_hub(hf_hub_id, private=True, token=hf_token)
    else:
        raise ValueError('Missing "model_name" or "dataset"')


if __name__ == "__main__":
    import fire

    fire.Fire(main)
