#!/bin/bash

#SBATCH -p mit_normal
#SBATCH --mincpus 96
#SBATCH --mem 90000
#SBATCH --time 360
#SBATCH --output runs/tokenize_data.log

/home/wyf/.conda/envs/torch/bin/python tokenize_data.py
