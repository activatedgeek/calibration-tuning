# LLM Calibration

## Setup

Create a new conda environment:
```shell
conda env create -f environment.yml -n <env>
```

Activate environment:
```shell
conda activate <env>
```

And finally, run:
```
pip install -e .
```

**NOTE**: If a different PyTorch CUDA compilation is required, use extra index repositories. e.g. For CUDA 11.8, run:
```shell
pip install --no-cache-dir -U torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

## Run

All arguments from the `main` method in each of the scripts below
qualify as command line arguments.

**NOTE**: Use `CUDA_VISIBLE_DEVICES` to limit the GPUs used.

### Fine-Tune

An example command to run fine-tuning with Llama2-7b:
```shell
./autotorchrun experiments/fine_tune.py \
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

### Evaluate

An example command for evaluation.

```shell
./autotorchrun experiments/evaluate.py --model_name=llama2_7b --dataset=eval:all
```

To evaluate for open-ended sequences:

```shell
./autotorchrun experiments/evaluate.py --model_name=llama2_7b --dataset=mmlu:business_ethics --mode=oe_substring --prompt_style=oe
```

with fuzzy matching (currently on GPT4, so requires setting OPENAI_API_KEY env var):

```shell
./autotorchrun experiments/evaluate.py --model_name=llama2_7b --dataset=mmlu:business_ethics --mode=oe_fuzzy_gpt-4-0613 --prompt_style=oe
```

## Details

### Offline Dataset Generation

To create a CSV dataset of open-ended generations,
```shell
./autotorchrun experiments/generate.py outputs \
    --model-name=llama2_7b \
    --peft-dir=</optional/path/to/checkpoint/dir> \
    --dataset=sub_200k_c \
    --batch-size=10 \
    --log-dir=</path/to/log_dir>
```

This command will generate CSV files under `<log-dir>/outputs`.

Use this path to further generate a CSV dataset of uncertainty labels on top of the open-ended generations, 
```shell
./autotorchrun experiments/generate.py outputs \
    --model-name=llama2_7b \
    --peft-dir=</optional/path/to/checkpoint/dir> \
    --dataset=offline \
    --data-dir=<log-dir>/outputs \
    --batch-size=10 \
    --strategy=substring \
    --log-dir=</path/to/log_dir>
```

Use `--strategy=fuzzy_gpt-3.5-turbo-1106` for generating labels via GPT 3.5 Turbo.

This command will generate CSV files under `<log-dir>/labels`. Note the `--dataset` and `--data-dir` arguments.

### Cache

Send `--use_dataset_cache=False` to rebuild [HuggingFace dataset cache](https://huggingface.co/docs/datasets/v2.14.4/en/cache#cache-files).

# LICENSE

Apache 2.0
