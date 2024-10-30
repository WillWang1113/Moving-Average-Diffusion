#!/bin/bash

# FactOnly + Frequency denoise
gpu=1
# batch_size=(256)
# batch_size=(128)
batch_size=(64)
model_config=MADtime_pl_FactOnly_SETKS_FreqDoi_CFG_puncond0.0
save_dir=/home/user/data/FrequencyDiffusion/savings
# save_dir=/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings

seq_len=96
# pred_len=(96)
pred_len=(96 192 336 720)
data_pth=(
  # etth1
  etth2
  exchange_rate
  electricity
  traffic
  weather
  # ettm1
  ettm2
)

# # Train
for bs in "${batch_size[@]}"; do
  for i in "${pred_len[@]}"; do
    for j in "${data_pth[@]}"; do
      echo $j $i
      init_model="${j}_PatchTST_sl96_pl${i}_maks4_predstats0"
      echo $init_model

      python -u scripts/train_pl.py \
        -dc $j \
        -mc $model_config \
        --save_dir $save_dir \
        --seq_len $seq_len \
        --pred_len $i \
        --gpu $gpu --num_train 5 --batch_size $bs
      # python scripts/sample_pl.py \
      #   --save_dir $save_dir \
      #   --dataset $j \
      #   --pred_len $i \
      #   --task S \
      #   --model_name "${model_config}_bs${bs}" \
      #   --deterministic \
      #   --n_sample 100 \
      #   --gpu $gpu \
      #   --num_train 5 \
      #   --w_cond 0.0 \
      #   --init_model $init_model \
      #   --start_ks 4
      python scripts/sample_pl.py \
        --save_dir $save_dir \
        --dataset $j \
        --pred_len $i \
        --task S \
        --model_name "${model_config}_bs${bs}" \
        --deterministic \
        --n_sample 100 \
        --gpu $gpu \
        --num_train 5 \
        --w_cond 1.0 \
        # --init_model $init_model \
        # --start_ks 4
    done
  done
done