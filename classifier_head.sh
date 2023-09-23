#!/bin/bash

#SBATCH --job-name=test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=8
#SBATCH --partition=ocp
#SBATCH -t 3-0

source /data/home/mshuaibi/miniconda3/bin/activate
conda activate llm-calibration
cd /data/home/ngruver/llm-calibration

ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

WANDB_MODE=online \
PYTHONPATH="/data/home/ngruver/llm-calibration:/data/home/ngruver/ocp-modeling-dev/llm/bitsandbytes" \
srun \ 
    --unbuffered \
    --output slurm/%j/%j_%t_log.out \
    --error slurm/%j/%j_%t_log.out \
    torchrun \
    --nnodes 2 \
    --nproc_per_node 8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$ADDR:29400" \
    experiments/feature_classifier_train.py \
    --model_name=llama2_7b \
    --dataset=combined_100k \
    --log_dir=.