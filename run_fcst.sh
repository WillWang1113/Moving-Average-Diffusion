#!/bin/bash

gpu=0
# model_config=(MADTC_NFD)
model_config=(MADTC_NFD_MLP_x0)
# model_config=(MADTC_NFD_MLP_x0 MADTC_NFD_MLP_eps MADTC_NFD_DIT_eps MADTC_NFD_DIT_x0)
# model_config=(MADTC_FD_lr1e-4 MADTC_FD MADTC_NFD_lr1e-4 MADTC_NFD)

save_dir=savings/MAD_fcst/savings
num_train=5
seq_len=96
# pred_len=(96)
pred_len=(96 192 336 720)
data_pth=(
  # etth1
  electricity
  etth2
  exchange_rate
  traffic
  weather
  # ettm1
  ettm2
)

# # Train
for mc in "${model_config[@]}"; do
  for i in "${pred_len[@]}"; do
    for j in "${data_pth[@]}"; do
      echo $j $i $mc

      python -u scripts/train_fcst.py \
        -dc $j \
        -mc $mc \
        --save_dir $save_dir \
        --seq_len $seq_len \
        --pred_len $i \
        --gpu $gpu --num_train $num_train --batch_size 256 --condition fcst

      python scripts/sample_fcst.py -dc $j \
        --model_name "${mc}_bs256_condfcst" \
        --num_train $num_train \
        --save_dir $save_dir \
        --condition fcst \
        --w_cond 1 \
        --n_sample 100 \
        --deterministic \
        --gpu $gpu \
        --seq_len $seq_len \
        --pred_len $i --fast_sample

    done
  done
done
