#!/bin/bash

gpu=1
# batch_size=(64)
model_config=(PatchTST)
maks=(6)
# model_config=(MADtime_FactOnly_SETKS_learnmean_freqdenoise_puncond0.5)
save_dir=/home/user/data/MAD/savings
# save_dir=/home/user/data/FrequencyDiffusion/savings
# save_dir=/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings

seq_len=96
pred_len=(96)
# pred_len=(96 192 336 720)
data_pth=(
  # etth1
  # etth2
  # exchange_rate
  # electricity
  traffic
  # weather
  # ettm1
  # ettm2
)

# # Train
for ks in "${maks[@]}"; do
  for mc in "${model_config[@]}"; do
    for i in "${pred_len[@]}"; do
      for j in "${data_pth[@]}"; do
        echo $j $i $mc
        init_model="${j}_${mc}_sl96_pl${i}_maks${ks}_predstats0"
        echo $init_model

        python scripts/LR_eval.py \
          --save_dir $save_dir \
          --dataset $j \
          --pred_len $i \
          --task S \
          --model_name "${mc}_bs64" \
          --gpu $gpu \
          --num_train 1 \
          --init_model $init_model 
      done
    done
  done
done
