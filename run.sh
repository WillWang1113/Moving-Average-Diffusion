#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
# pred_len=(96)
save_dir=/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings
# save_dir=/mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings_newcond
pred_len=(96 192 336 720)
data_pth=(
  etth1
  etth2
  ettm1
  ettm2
  electricity
  exchange_rate
  traffic
  weather
)

# Train
for i in "${pred_len[@]}"; do
  for j in "${data_pth[@]}"; do
    echo $j $i
    python -u scripts/train.py \
      -dc $j \
      -mc MADfreq \
      --save_dir $save_dir \
      --seq_len 96 \
      --pred_len $i
  done
done

# Test
for i in "${pred_len[@]}"; do
  for j in "${data_pth[@]}"; do
    echo $j $i
    python scripts/sample.py \
      --save_dir $save_dir \
      --dataset $j \
      --pred_len $i \
      --task S \
      --model_name MADFreq \
      --kind freq \
      --deterministic \
      --fast_sample \
      --n_sample 100
  done
done
# python scripts/sample.py --model_name MADFreq --deterministic --fast_sample --collect_all --kind freq --n_sample 100
# python scripts/sample.py --model_name MADTime --deterministic --fast_sample --collect_all --kind time --n_sample 100
