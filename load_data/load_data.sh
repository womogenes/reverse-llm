#!/bin/bash

#SBATCH -p mit_normal
#SBATCH --mincpus 64
#SBATCH --mem 256000

module load miniforge

/home/wyf/.conda/envs/torch/bin/python load_data.py
