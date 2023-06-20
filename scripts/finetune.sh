#!/usr/bin/env bash

__PORT=$(shuf -i 10000-65500 -n 1)

accelerate launch --multi_gpu --main_process_port=${__PORT} \
experiments/finetune.py \
    --model-name=seq_open_llama_13b \
    --dataset=yelp