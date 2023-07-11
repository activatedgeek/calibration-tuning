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
python experiments/evaluate_gen.py \
    --seed=137 \
    --model-name=open_llama_13b \
    --dataset=mmlu \
    --dataset-instance=business_ethics
```

### Fine-Tuning

An example command to run fine-tuning on the Alpaca dataset:
```shell
./scripts/finetune.sh --seed=137 --epochs=1
```

# LICENSE

Apache 2.0