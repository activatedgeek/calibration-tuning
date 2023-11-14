# LLM Calibration

## Setup

Create a new conda environment (if needed):
```
conda env create -f environment.yml -n <env_name>
```

Install CUDA-compiled PyTorch version from [here](https://pytorch.org). The codebase
has been tested with PyTorch version `2.0`.
```shell
pip install 'torch>=2.0' torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

And finally, run
```
pip install -e .
```

## Run

**NOTE**: Use `CUDA_VISIBLE_DEVICES` to limit the GPUs used.

### Evaluate

An example command for evaluation.

```shell
./autotorchrun experiments/evaluate.py --model_name=llama2_7b --dataset=mmlu:business_ethics
```

To evaluate for open-ended sequences:

```shell
./autotorchrun experiments/evaluate.py --model_name=llama2_7b --dataset=mmlu:business_ethics --mode=oe_substring --prompt_style=oe
```

with fuzzy matching (currently on GPT4, so requires setting OPENAI_API_KEY env var):

```shell
./autotorchrun experiments/evaluate.py --model_name=llama2_7b --dataset=mmlu:business_ethics --mode=oe_fuzzy_gpt4 --prompt_style=oe
```


### Fine-Tuning

An example command to run fine-tuning with Llama2-7b:
```shell
./autotorchrun experiments/finetune.py --model_name=llama2_7b --dataset=mmlu:business_ethics
```

## Details

### Cache

Send `--use_dataset_cache=False` to rebuild [HuggingFace dataset cache](https://huggingface.co/docs/datasets/v2.14.4/en/cache#cache-files).

# LICENSE

Apache 2.0