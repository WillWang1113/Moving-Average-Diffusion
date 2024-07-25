#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1
gpu=1
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
data_pth=(
  etth1
  # etth2
  # ettm1
  # ettm2
  # traffic
  # electricity
  # exchange_rate
  # weather
)

# Train
for i in "${pred_len[@]}"; do
  for j in "${data_pth[@]}"; do
    echo $j $i
    python -u scripts/train_gts.py \
      -dc $j \
      -mc $model_config \
      --save_dir $save_dir \
      --seq_len $seq_len \
      --pred_len $i \
      --gpu $gpu --num_train 1 --dataset electricity_nips
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
