# LLM Calibration

## Setup

Create a new conda environment (if needed):
```
conda env create -f environment.yml -n <env_name>
```

Install CUDA-compiled PyTorch version from [here](https://pytorch.org). The codebase
has been tested with PyTorch version `1.13`.
```shell
pip install 'torch<2.0' torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

And finally, run
```
pip install -e .
```

## Run

Set `CUDA_VISIBLE_DEVICES` to limit the number of GPUs used.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/finetune.sh --epochs=1 <...args>
```

**NOTE**: If `CUDA_VISIBLE_DEVICES` is not set, then all GPUs available will be used.

# LICENSE

Apache 2.0