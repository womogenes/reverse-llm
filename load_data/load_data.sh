#!/bin/bash

#SBATCH -p mit_normal
#SBATCH --mincpus 32
#SBATCH --mem 185000
#SBATCH --time 360

/home/wyf/.conda/envs/torch/bin/python load_ultrachat_data.py
