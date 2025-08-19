#!/bin/bash

#SBATCH -p mit_normal_gpu
#SBATCH --gres gpu:h200:2
#SBATCH --mincpus 32
#SBATCH --mem 185000
#SBATCH --time 360
#SBATCH --output runs_gpt2/train_005.log

/home/wyf/.conda/envs/torch/bin/accelerate launch train.py
