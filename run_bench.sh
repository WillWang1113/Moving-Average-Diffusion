#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
model_name=PatchTST

# python -u scripts/train_bench_THU.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /home/user/data/THU-timeseries/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_96 \
#   --model $model_name \
#   --data ETTh1 \
#   --features S \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1

# python -u scripts/train_bench_THU.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /home/user/data/THU-timeseries/electricity/ \
#   --data_path electricity.csv \
#   --model_id electricity_96_192 \
#   --model $model_name \
#   --data custom \
#   --features S \
#   --seq_len 96 \
#   --num_workers 1\
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --itr 1

python -u scripts/train_bench_THU.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/user/data/THU-timeseries/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --num_workers 1 \
  --des 'Exp' \
  --n_heads 4 \
  --itr 1
# python -u scripts/train_bench_THU.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /home/user/data/THU-timeseries/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_336 \
#   --model $model_name \
#   --data ETTh1 \
#   --features S \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1

# python -u scripts/train_bench_THU.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /home/user/data/THU-timeseries/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_720 \
#   --model $model_name \
#   --data ETTh1 \
#   --features S \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1

# export CUDA_VISIBLE_DEVICES=1
# pred_len=(96 192 336 720)
# data_pth=('electricity/electricity_NF.csv'
#   'ETT-small/ETTh1_NF.csv'
#   'ETT-small/ETTh2_NF.csv'
#   'ETT-small/ETTm1_NF.csv'
#   'ETT-small/ETTm2_NF.csv'
#   'exchange_rate/exchange_rate_NF.csv'
#   'weather/weather_NF.csv')

# for i in "${pred_len[@]}"; do
#   for j in "${data_pth[@]}"; do
#     echo $j $i
#     python -u scripts/train_bench_NF.py \
#       --task U \
#       --root_pth /home/user/data/THU-timeseries \
#       --data_pth $j \
#       --seq_len 96 \
#       --pred_len $i
#   done
# done

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

