#!/usr/bin/env bash

__PORT=$(shuf -i 10000-65500 -n 1)

accelerate launch --multi_gpu --main_process_port=${__PORT} \
experiments/evaluate_gen.py \
    --seed=137 \
    --model-name=open_llama_13b \
    --dataset=truthful_qa_mc1 \
    "${@}"
