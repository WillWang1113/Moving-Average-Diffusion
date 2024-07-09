#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1

model_name=DLinear

python -u scripts/train_benchmark.py \
  --task_name long_term_forecast \
  --checkpoints /home/user/data/Benchmarks/ \
  --is_training 1 \
  --root_path /home/user/data/THU-timeseries/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 2