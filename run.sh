#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1
gpu=0
model_config=MADtime_pl
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
# pred_len=(96)
pred_len=(96 192 336 720)
data_pth=(
  etth1
  etth2
  ettm1
  ettm2
  electricity
  traffic
  exchange_rate
  weather
  # mfred
)

# Train
for i in "${pred_len[@]}"; do
  for j in "${data_pth[@]}"; do
    echo $j $i
    python -u scripts/train_pl.py \
      -dc $j \
      -mc $model_config \
      --save_dir $save_dir \
      --seq_len $seq_len \
      --pred_len $i \
      --gpu $gpu --num_train 5
    python scripts/sample_pl.py \
      --save_dir $save_dir \
      --dataset $j \
      --pred_len $i \
      --task S \
      --model_name $model_config \
      --kind time \
      --deterministic \
      --n_sample 100 \
      --gpu $gpu --num_train 5

  done
done



seq_len=288
pred_len=(288 432)
data_pth=(
  # etth1
  # etth2
  # ettm1
  # ettm2
  # electricity
  # traffic
  # exchange_rate
  # weather
  mfred
)


# Train
for i in "${pred_len[@]}"; do
  for j in "${data_pth[@]}"; do
    echo $j $i
    python -u scripts/train_pl.py \
      -dc $j \
      -mc $model_config \
      --save_dir $save_dir \
      --seq_len $seq_len \
      --pred_len $i \
      --gpu $gpu --num_train 5
    python scripts/sample_pl.py \
      --save_dir $save_dir \
      --dataset $j \
      --pred_len $i \
      --task S \
      --model_name $model_config \
      --kind time \
      --deterministic \
      --n_sample 100 \
      --gpu $gpu --num_train 5

  done
done
