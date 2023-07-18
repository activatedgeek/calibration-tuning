#!/usr/bin/env bash

__PORT=$(shuf -i 10000-65500 -n 1)

SRC_DIR=$(dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")

LOG_DIR=${SRC_DIR}/.log/finetune-local

accelerate launch --multi_gpu --main_process_port=${__PORT} \
experiments/finetune.py \
    --log_dir=${LOG_DIR} \
    --model_name=open_llama_7b \
    --dataset=mmlu \
    --dataset_instance=business_ethics \
    "${@}"
