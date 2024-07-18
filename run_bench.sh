#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# NIPS DATASETS
# pred_len=(96 192 336 720)
# data_pth=(
#   "ETTh1"
#   "ETTh2"
#   "ETTm1"
#   "ETTm2"
#   "ECL"
#   "Weather"
#   "TrafficL"
#   "Exchange"
# )

# for i in "${pred_len[@]}"; do
#   for j in "${data_pth[@]}"; do
#     echo $j $i
#     python -u scripts/train_bench_NF.py \
#       --task U \
#       --data_dir /mnt/ExtraDisk/wcx/research \
#       --save_dir /mnt/ExtraDisk/wcx/research/benchmarks \
#       --dataset $j \
#       --seq_len 96 \
#       --pred_len $i
#   done
# done

pred_len=(288 432 576)
data_pth=

for i in "${pred_len[@]}"; do
  python -u scripts/train_bench_NF.py \
    --task U \
    --data_dir /mnt/ExtraDisk/wcx/research/FrequencyDiffusion/dataset \
    --save_dir /mnt/ExtraDisk/wcx/research/benchmarks \
    --dataset MFRED \
    --seq_len 288 \
    --pred_len $i
done
