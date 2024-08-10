#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# NIPS DATASETS
# pred_len=(96)
# pred_len=(96 192 288)
pred_len=(96 192 336 720)
DATASETS=(
  'electricity_nips'
  'traffic_nips'
  'solar_nips'
  'exchange_rate_nips'
  'wiki2000_nips'
  'kdd_cup_2018_without_missing'
)
data_dir=
for i in "${pred_len[@]}"; do
  for j in "${DATASETS[@]}"; do
    echo $j $i
    python -u scripts/train_bench_gts.py \
      --save_dir /home/user/data/GTS_benchmark \
      --dataset $j \
      --seq_len 96 \
      --pred_len $i 
  done
done

# pred_len=(288 432 576)
# DATASETS=

# for i in "${pred_len[@]}"; do
#   python -u scripts/train_bench_NF.py \
#     --task S \
#     --data_dir /mnt/ExtraDisk/wcx/research/FrequencyDiffusion/dataset \
#     --save_dir /mnt/ExtraDisk/wcx/research/benchmarks \
#     --dataset MFRED \
#     --seq_len 288 \
#     --pred_len $i
# done
