import wandb
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from peft import PeftModel

from llm.datasets import get_dataset, get_loader
from llm.datasets.llm_utils import LMText, prepare_batch, DataCollatorForSupervisedDataset
from llm.models import get_model
from llm.models.peft import get_lora_model
from llm.logging import entrypoint


def generate_outputs_main(
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    num_workers=8,
    batch_size=1,
    model_name=None,
    model_dir=None,
    peft_dir=None,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    use_dataset_cache=True,
    prompt_style="oe",
    max_new_tokens=10,
):
    accelerator = Accelerator()

    config = {
        "seed": seed,
        "model_name": model_name,
        "model_dir": model_dir,
        "peft_dir": peft_dir,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "prompt_style": prompt_style,
        "max_new_tokens": max_new_tokens,
    }
    if accelerator.is_main_process:
        wandb.config.update(config)

    tokenizer = get_model(
        f"{model_name}_tokenizer",
        model_dir=model_dir,
    )

    model = get_model(
        model_name,
        device_map={"": accelerator.local_process_index},
        torch_dtype=torch.float16,
        model_dir=model_dir,
        use_cache=False,
        tokenizer=tokenizer,
        load_in_8bit=True,
    )

    model = get_lora_model(
        model,
        peft_dir=peft_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        is_trainable=False,
        adapter_name="default",
    )

    model.eval()

    with accelerator.main_process_first():
        train_data, _, _ = get_dataset(
            dataset,
            root=data_dir,
            tokenizer=tokenizer,
            seed=seed,
            num_workers=num_workers,
            use_cache=use_dataset_cache,
            prompt_style=prompt_style,
        )
    
    train_loader = get_loader(
        train_data,
        batch_size=batch_size,
        pin_memory=True,
        accelerator=accelerator,
    )
    
    collate_fn = DataCollatorForSupervisedDataset(tokenizer)

    for raw_inputs in tqdm(train_loader):
        ## Skip "target" for generation.
        model_inputs = {k: v for k,v in raw_inputs.items() if k != "target"}
        model_inputs = prepare_batch(tokenizer, model_inputs, prompt_style=prompt_style)

        model_inputs = collate_fn(model_inputs)
        model_inputs = {k: v.to(accelerator.device) for k, v in model_inputs.items()}

        if isinstance(model, PeftModel):
            model.set_adapter("default")

        model_outputs = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

        raw_model_inputs = [
            str(LMText(**{**dict(zip(raw_inputs.keys(), vals)), "target": ""}))
            for vals in zip(*raw_inputs.values())
        ]

        raw_model_outputs = tokenizer.batch_decode(model_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        raw_outputs = [t[len(s):] for s, t in zip(raw_model_inputs, raw_model_outputs)]

        breakpoint()


if __name__ == "__main__":
    import fire

    fire.Fire(dict(outputs=entrypoint(generate_outputs_main)))
