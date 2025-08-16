#!/bin/bash

#SBATCH -p mit_normal_gpu
#SBATCH --mincpus 32
#SBATCH --mem 64000
#SBATCH --output runs/tokenize.log
#SBATCH --time 360 

#/home/wyf/.conda/envs/torch/bin/accelerate launch train_fineweb.py
/home/wyf/.conda/envs/torch/bin/python tokenize_data.py

