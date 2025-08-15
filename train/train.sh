#!/bin/bash

#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --mincpus 32
#SBATCH --mem 64000
#SBATCH --output runs/fineweb_2_000_000_run_014.log
#SBATCH --time 360 

/home/wyf/.conda/envs/torch/bin/accelerate launch train_fineweb.py
