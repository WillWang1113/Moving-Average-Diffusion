#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1
gpu=0
model_config=MADfreq_gts
save_dir=/home/user/data/FrequencyDiffusion/savings
# save_dir=/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings

# ! MFRED
# seq_len=288
# # pred_len=(288 432 576)
# pred_len=(288)
# data_pth=(
#   mfred
# )

# ! Others
seq_len=96
pred_len=(96)
# pred_len=(96 192 336 720)
DATASETS=(
  'electricity_nips'
  # 'traffic_nips'
  # 'solar_nips'
  # 'exchange_rate_nips'
  # 'wiki2000_nips'
  # 'kdd_cup_2018_without_missing'
)

# Train
for i in "${pred_len[@]}"; do
  for j in "${DATASETS[@]}"; do
    echo $j $i
    python -u scripts/train_gts.py \
      -mc $model_config \
      --save_dir $save_dir \
      --seq_len $seq_len \
      --pred_len $i \
      --gpu $gpu --num_train 1 --dataset $j --num_samples 100
    # python scripts/sample_pl.py \
    #   --save_dir $save_dir \
    #   --dataset $j \
    #   --pred_len $i \
    #   --task S \
    #   --model_name $model_config \
    #   --kind freq \
    #   --deterministic \
    #   --n_sample 100 \
    #   --gpu $gpu --fast_sample --num_train 1

  done
done
