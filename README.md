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
```shell
pip install -e .
```

This will install the [`llm`](./llm) package.

**NOTE**: If a different PyTorch CUDA compilation is required, use extra index repositories. e.g. For CUDA 11.8, run:
```shell
pip install --no-cache-dir -U torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

## Usage

All arguments from the `main` method in each of the scripts below
qualify as command line arguments.

**Environment Variables**:

- `HF_HOME`: Path to directory where HuggingFace assets (models and datasets) are cached.
- `OPENAI_API_KEY`: OpenAI API key. Used for labeling a generated dataset and evaluations only.
- `CUDA_VISIBLE_DEVICES`: Use to limit the GPU visibility used by the scripts.

### Dataset Generation

**NOTE**: The Story Cloze dataset (2018 version) requires manual download. See instructions [here](https://cs.rochester.edu/nlp/rocstories/). After getting the CSV file, place it at `${HF_HOME}/datasets/story_cloze/2018`.

#### Output Generation
To create a CSV dataset of open-ended generations at `<outputs-log-dir>/outputs`.
`<outputs-log-dir>` is auto-generated or can be explicitly specified using `--log-dir` argument.

```shell
python experiments/generate.py outputs --dataset=all_20k_uniform --prompt-style=oe --model-name=llama2_13b_chat --max-new-tokens=30 --batch-size=16 --kshot=1
```

For multiple-choice generations,

```shell
python experiments/generate.py outputs --dataset=all_20k_uniform --prompt-style=choice --model-name=llama2_13b_chat --max-new-tokens=1 --batch-size=16
```

#### Uncertainty Query Label Generation

To generate the dataset with uncertainty query labels at `<labels-log-dir>/labels` (auto-generated or specified via `--log-dir`), 

```shell
python experiments/generate.py labels --dataset=offline:<outputs-log-dir>/outputs --model-name=llama2_13b_chat --strategy=substring --batch-size=16
```

Use `--strategy=fuzzy_gpt-3.5-turbo-1106` for generating labels via GPT 3.5 Turbo.

### Calibration-Tune

An example command to run fine-tuning with Llama2-7b:
```shell
./autotorchrun experiments/calibration_tune.py \
    --model_name=llama2_7b \
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

# LICENSE

Apache 2.0
