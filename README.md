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