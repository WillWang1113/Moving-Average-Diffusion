#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
# pred_len=(720)
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

for i in "${pred_len[@]}"; do
  for j in "${data_pth[@]}"; do
    echo $j $i
    python -u scripts/train.py \
      -dc $j \
      -mc MADfreq \
      --save_dir /mnt/ExtraDisk/wcx/research/FrequencyDiffusion/savings \
      --seq_len 96 \
      --pred_len $i
  done
done

# python scripts/sample.py --model_name MADFreq --deterministic --fast_sample --collect_all --kind freq --n_sample 100
# python scripts/sample.py --model_name MADTime --deterministic --fast_sample --collect_all --kind time --n_sample 100
