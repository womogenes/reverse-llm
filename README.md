# Resource usage:

salloc -p mit_normal_gpu \
    --gres=gpu:h100:2 \
    --time=360 \
    --cpus-per-gpu=4 \
    --mem=32000

cpus are being maxed out though
