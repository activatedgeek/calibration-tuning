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
./experiments/evaluate.sh --model_name=llama2_7b --dataset=mmlu:business_ethics
```

### Fine-Tuning

An example command to run fine-tuning on the Alpaca dataset with Llama2-7b:
```shell
./experiments/finetune.sh --model_name=llama2_7b --dataset=alpaca
```

## Details

### Cache

Send `use_cache=False` in the `get_dataset` function to rebuild [HuggingFace cache](https://huggingface.co/docs/datasets/v2.14.4/en/cache#cache-files).

# LICENSE

Apache 2.0