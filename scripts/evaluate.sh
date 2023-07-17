#!/usr/bin/env bash

__PORT=$(shuf -i 10000-65500 -n 1)

accelerate launch --multi_gpu --main_process_port=${__PORT} \
experiments/evaluate.py \
    --model-name=open_llama_7b \
    --dataset=mmlu \
    --dataset-instance=business_ethics \
    "${@}"
    