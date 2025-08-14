#!/bin/bash

#SBATCH -p mit_normal
#SBATCH --mincpus 16
#SBATCH --mem 32000

/home/wyf/.conda/envs/torch/bin/python load_data.py
