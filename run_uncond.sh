#!/bin/bash

gpu=0
model_config=(MATSD_uncond)

save_dir=/home/user/data2/ICML_camera_ready_test/uncond
num_train=1
# pred_len=24 means generating L=24 time series
pred_len=(24)
# seq_len=0 means no look-back window, because unconditional generation instead of forecasting
seq_len=0
n_sample=1
data_pth=(
  exchange_rate
  etth2
)

# Training and sampling for etth2 and exchange
for mc in "${model_config[@]}"; do
  for i in "${pred_len[@]}"; do
    for j in "${data_pth[@]}"; do
      echo $j $i $mc

      python -u scripts/train_uncond.py \
        -dc $j \
        -mc $mc \
        --save_dir $save_dir \
        --pred_len $i \
        --seq_len $seq_len \
        --gpu $gpu --num_train $num_train --batch_size 64

      python scripts/rebuttal_sample_uncond.py -dc $j \
        --model_name $mc \
        --num_train $num_train \
        --save_dir $save_dir \
        --w_cond 0 \
        --n_sample $n_sample \
        --deterministic \
        --gpu $gpu \
        --seq_len $seq_len \
        --pred_len $i

    done
  done
done

# Training and sampling for ecg
python -u scripts/train_uncond_ecg.py \
  -dc ecg \
  -mc DDPM_IN_MA \
  --save_dir $save_dir \
  --pred_len 24 \
  --seq_len $seq_len \
  --gpu $gpu --num_train $num_train --batch_size 64

python scripts/sample_uncond_ecg.py -dc ecg \
  --model_name DDPM_IN_MA \
  --num_train $num_train \
  --save_dir $save_dir \
  --w_cond 0 \
  --n_sample $n_sample \
  --deterministic \
  --gpu $gpu \
  --seq_len $seq_len \
  --pred_len 24

# final evaluation
python scripts/evaluate_uncond.py
