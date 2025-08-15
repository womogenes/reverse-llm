#!/bin/bash

#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:a100:4
#SBATCH --mincpus 64
#SBATCH --mem 32000
#SBATCH --output runs/fineweb_200_000_run_009.log
#SBATCH --requeue

/home/wyf/.conda/envs/torch/bin/accelerate launch train_fineweb.py
