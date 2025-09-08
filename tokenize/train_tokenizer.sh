#!/bin/bash

#SBATCH --mincpus 10
#SBATCH --mem 84000
#SBATCH --time 360
#SBATCH --output runs/train_tokenizer.log

/home/wyf/.conda/envs/torch/bin/python train_tokenizer.py
