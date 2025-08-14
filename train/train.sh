#!/bin/bash

#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --mincpus 16
#SBATCH --mem 64000
#SBATCH --output runs/fineweb_200_000_run_005_continued_001.log

/home/wyf/.conda/envs/torch/bin/accelerate launch train_fineweb.py
