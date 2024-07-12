#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
# pred_len=(720)
pred_len=(96 192 336 720)
data_pth=(
  "ETTh1"
  "ETTh2"
  "ETTm1"
  "ETTm2"
  "ECL"
  "Weather"
  "TrafficL"
  "Exchange"
)

for i in "${pred_len[@]}"; do
  for j in "${data_pth[@]}"; do
    echo $j $i
    python -u scripts/train_bench_NF.py \
      --task U \
      --data_dir /mnt/ExtraDisk/wcx/research \
      --save_dir /mnt/ExtraDisk/wcx/research/benchmarks \
      --dataset $j \
      --seq_len 96 \
      --pred_len $i
  done
done

# for i in 288 576 864; do
#   echo $i

#   python -u scripts/train_bench_NF.py \
#     --task U \
#     --root_pth /home/user/data/THU-timeseries \
#     --data_pth MFRED/MFRED_NF.csv \
#     --seq_len 288 \
#     --pred_len $i

# done

# THU
