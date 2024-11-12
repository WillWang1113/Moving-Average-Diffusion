#!/bin/bash

gpu=1
# batch_size=(64)
# model_config=(MADtime_FactOnly_SETKS_learnmean_freqdenoise MADtime_FactOnly_SETKS_learnmean_freqdenoise_puncond0.5)
# model_config=(MADfreq_FactOnly_learnmean MADtime_FactOnly_SETKS_learnmean_freqdenoise)
# model_config=(MADtime_naive MADtime_learnmean MADfreq_naive MADfreq_learnmean MADtime_learnmean_freqdenoise MADtime_FactOnly_SETKS_learnmean_freqdenoise MADtime_FactOnly_SETKS_learnmean_freqdenoise_puncond0.5)
model_config=(MADfreq_puncond0.5)
# model_config=(MADfreq_puncond0.5 MADfreq_FactOnly_puncond0.5)
# model_config=(MADfreq MADfreq_FactOnly MADfreq_puncond0.5 MADfreq_FactOnly_puncond0.5)
# model_config=(MADtime_FactOnly_SETKS_learnmean_freqdenoise_puncond0.5)
# model_config=MADtime_pl_FactOnly_SETKS_FreqDoi_CFG_puncond0.2
save_dir=/home/user/data/MAD/savings
# save_dir=/home/user/data/FrequencyDiffusion/savings
# save_dir=/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings

seq_len=96
pred_len=(96)
# pred_len=(96 192 336 720)
data_pth=(
  # etth1
  etth2
  # exchange_rate
  # electricity
  # traffic
  # weather
  # ettm1
  # ettm2
)

# # Train
for mc in "${model_config[@]}"; do
  for i in "${pred_len[@]}"; do
    for j in "${data_pth[@]}"; do
      echo $j $i $mc
      init_model="${j}_PatchTST_sl96_pl${i}_maks4_predstats0"
      echo $init_model

      # python -u scripts/train_pl.py \
      #   -dc $j \
      #   -mc $mc \
      #   --save_dir $save_dir \
      #   --seq_len $seq_len \
      #   --pred_len $i \
      #   --gpu $gpu --num_train 5 --batch_size 64

      # python scripts/sample_pl.py \
      #   --save_dir $save_dir \
      #   --dataset $j \
      #   --pred_len $i \
      #   --task S \
      #   --model_name "${mc}_bs64" \
      #   --deterministic \
      #   --n_sample 100 \
      #   --gpu $gpu \
      #   --num_train 5 \
      #   --w_cond 0.0 \
      #   --init_model $init_model \
      #   --start_ks 4

      python scripts/sample_pl_uncond.py \
        --save_dir $save_dir \
        --dataset $j \
        --pred_len $i \
        --task S \
        --model_name "${mc}_bs64" \
        --deterministic \
        --n_sample 5 \
        --gpu $gpu \
        --num_train 5 \
        --w_cond 0.0 --fast_sample
    done
  done
done
