#!/usr/bin/env bash

__PORT=$(shuf -i 10000-65500 -n 1)

accelerate launch --multi_gpu --main_process_port=${__PORT} \
experiments/finetune.py \
    --model_name=llama2_7b \
    --model_dir=${MODELDIR}/models--meta-llama--Llama-2-7b \
    --dataset=mmlu \
    --dataset_instance=business_ethics \
    "${@}"
