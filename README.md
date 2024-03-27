# LLM Calibration

## HuggingFace Release

We release the following calibration-tuned models as [PEFT](https://huggingface.co/docs/peft) adapters via HuggingFace.

<table>
  <tr>
    <td rowspan=6 valign="center">Open-Ended Generation</td>
    <td valign="top">Llama 2 7B</td>
    <td valign="top"><a href="https://huggingface.co/calibration-tuning/Llama-2-7b-hf-ct-oe" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg"/></a></td>
  </tr>
  <tr>
    <td valign="top">Llama 2 7B Chat</td>
    <td valign="top"><a href="https://huggingface.co/calibration-tuning/Llama-2-7b-chat-hf-ct-oe" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg"/></a></td>
  </tr>
  <tr>
    <td valign="top">Llama 2 13B</td>
    <td valign="top"><a href="https://huggingface.co/calibration-tuning/Llama-2-13b-hf-ct-oe" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg"/></a></td>
  </tr>
  <tr>
    <td valign="top">Llama 2 13B Chat</td>
    <td valign="top"><a href="https://huggingface.co/calibration-tuning/Llama-2-13b-chat-hf-ct-oe" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg"/></a></td>
  </tr> 
  <tr>
    <td valign="top">Mistral 7B</td>
    <td valign="top"><a href="https://huggingface.co/calibration-tuning/Mistral-7B-v0.1-ct-oe" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg"/></a></td>
  </tr>  
  <tr>
    <td valign="top">Mistral 7B Instruct</td>
    <td valign="top"><a href="https://huggingface.co/calibration-tuning/Mistral-7B-Instruct-v0.2-ct-oe" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg"/></a></td>
  </tr>
  <tr>
    <td rowspan=6 valign="center">Multiple-Choice Question-Answering</td>
    <td valign="top">Llama 2 7B</td>
    <td valign="top"><a href="https://huggingface.co/calibration-tuning/Llama-2-7b-hf-ct-choice" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg"/></a></td>
  </tr>
  <tr>
    <td valign="top">Llama 2 7B Chat</td>
    <td valign="top"><a href="https://huggingface.co/calibration-tuning/Llama-2-7b-chat-hf-ct-choice" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg"/></a></td>
  </tr>
  <tr>
    <td valign="top">Llama 2 13B</td>
    <td valign="top"><a href="https://huggingface.co/calibration-tuning/Llama-2-13b-hf-ct-choice" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg"/></a></td>
  </tr>
  <tr>
    <td valign="top">Llama 2 13B Chat</td>
    <td valign="top"><a href="https://huggingface.co/calibration-tuning/Llama-2-13b-chat-hf-ct-choice" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg"/></a></td>
  </tr> 
  <tr>
    <td valign="top">Mistral 7B</td>
    <td valign="top"><a href="https://huggingface.co/calibration-tuning/Mistral-7B-v0.1-ct-choice" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg"/></a></td>
  </tr>  
  <tr>
    <td valign="top">Mistral 7B Instruct</td>
    <td valign="top"><a href="https://huggingface.co/calibration-tuning/Mistral-7B-Instruct-v0.2-ct-choice" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg"/></a></td>
  </tr>  
</table>

See [experiments/play.py](./experiments/play.py) for an example script of how to load and use the models.

## Development Setup

Create a new conda environment,

```shell
conda env create -f environment.yml -n <env>
```

Activate environment,

```shell
conda activate <env>
```

And finally run,

```shell
pip install -e .
```

This will install the [`llm`](./llm) package.

**NOTE**: If a different PyTorch CUDA compilation is required, use extra index repositories. e.g. For CUDA 11.8 run,

```shell
pip install --no-cache-dir -U torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

## Usage

All arguments from the `main` method in each of the scripts below
qualify as command line arguments.

**Environment Variables**:

- `HF_HOME`: Path to directory where HuggingFace assets (models and datasets) are cached.
- `OPENAI_API_KEY`: OpenAI API key. Used for labeling a generated dataset and evaluations only.
- `CUDA_VISIBLE_DEVICES`: Limit the GPU visibility used by the scripts.

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

### Training 

Checkpoints will be saved in an auto-generated directory, or can be configured via `--log-dir`.

#### CT-Query

To use the labeled dataset for calibration-tuning (`CT-Query`),

```shell
torchrun --nnodes=1 --nproc_per_node=auto experiments/calibration_tune.py --dataset=offline_noprompt:<labels-log-dir>/labels --model_name=llama2_13b_chat --batch-size=4 --kl-decay=1.0 --max-steps=5000
```

Use `--scale-temp` for temperature scaling of the uncertainty query predictions.

For other CLI arguments, see the `main` function of [experiments/calibration_tune.py](./experiments/calibration_tune.py).

#### CT-Probe / CT-LoRA

To use the labeled dataset for training a classifier head (`CT-Probe`),

```shell
torchrun --nnodes=1 --nproc_per_node=auto experiments/classifier_tune.py --dataset=offline_noprompt:<labels-log-dir>/labels --model_name=llama2_13b_chat --batch-size=4 --max-steps=5000
```

Use `--scale-temp` for temperature scaling of the classifier. 
Use `--with-lora` to enable trainable LoRA parameters.

For other CLI arguments, see the `main` function of [experiments/classifier_tune.py](./experiments/calibration_tune.py).

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
