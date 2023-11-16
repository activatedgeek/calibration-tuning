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

All arguments from the `main` method in each of the scripts below
qualify as command line arguments.

**NOTE**: Use `CUDA_VISIBLE_DEVICES` to limit the GPUs used.

### Evaluate

An example command for evaluation.

```shell
./autotorchrun experiments/evaluate.py --model_name=llama2_7b --dataset=eval:all
```

### Fine-Tune

An example command to run fine-tuning with Llama2-7b:
```shell
./autotorchrun experiments/finetune.py \
    --model_name=llama2_7b \
    --peft-dir=</optional/path/to/checkpoint/dir> \
    --dataset=sub_200k \
    --max-steps=10000
```

### Uncertainty-Tune

An example command to run fine-tuning with Llama2-7b:
```shell
./autotorchrun experiments/uncertainty_tune.py \
    --model_name=llama2_7b \
    --peft-dir=</optional/path/to/checkpoint/dir> \
    --dataset=sub_200k_c \
    --max-steps=10000
```

## Details

### Cache

Send `--use_dataset_cache=False` to rebuild [HuggingFace dataset cache](https://huggingface.co/docs/datasets/v2.14.4/en/cache#cache-files).

# LICENSE

Apache 2.0