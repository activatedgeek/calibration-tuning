#!/usr/bin/env bash

__NNODES=${NNODES:-1}
__NPROC_PER_NODE=${NPROC_PER_NODE:-$(python -c 'import torch; print(torch.cuda.device_count())')}
__IP=${IP:-$(hostname)}
__PORT=${PORT:-$(shuf -i 2000-65000 -n 1)}

torchrun --nnodes=${__NNODES} \
         --nproc_per_node=${__NPROC_PER_NODE} \
         --rdzv_endpoint=${__IP}:${__PORT} \
            experiments/finetune.py \
                --model_name=llama2_7b \
                --dataset=mmlu \
                --dataset_instance=business_ethics \
                "${@}"
