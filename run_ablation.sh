#!/bin/bash

gpu=1
model_config=(DDPM_norm_MA_stdsched)
# ratio_diff_steps=(1.0 0.9 0.7 0.5 0.3 0.1)
ratio_diff_steps=(0.25)
# model_config=(DDPM_MA DDPM_MA_stdsched)
# model_config=(DDPM_norm_MA DDPM_norm_MA_stdsched)

save_dir=/home/user/data/MAD_ablation/savings
num_train=5
pred_len=(576)
seq_len=0
n_sample=10
# pred_len=(96)
# pred_len=(96 192 336 720)
data_pth=(
  # etth1
  # weather
  # electricity
  # etth2
  # exchange_rate
  # traffic
  # ettm1
  # ettm2
  wind
  mfred
  solar
)

# # Train
for rds in "${ratio_diff_steps[@]}"; do
  for mc in "${model_config[@]}"; do
    for i in "${pred_len[@]}"; do
      for j in "${data_pth[@]}"; do
        echo $j $i $mc

        # python -u scripts/train_pl_ablation.py \
        #   -dc $j \
        #   -mc $mc \
        #   --save_dir $save_dir \
        #   --pred_len $i \
        #   --seq_len $seq_len \
        #   --gpu $gpu --num_train $num_train --batch_size 64

        python scripts/sample_pl_new_c_diffstep_ablation.py -dc $j \
          --model_name $mc \
          --num_train $num_train \
          --save_dir $save_dir \
          --w_cond 0 \
          --n_sample $n_sample \
          --deterministic \
          --gpu $gpu \
          --seq_len $seq_len \
          --pred_len $i --num_diff_steps $rds

      done
    done
  done
done

# # # Train
# for mc in "${model_config[@]}"; do
#   for i in "${pred_len[@]}"; do
#     for j in "${data_pth[@]}"; do
#       echo $j $i $mc

#       # python -u scripts/train_pl_ablation.py \
#       #   -dc $j \
#       #   -mc $mc \
#       #   --save_dir $save_dir \
#       #   --pred_len $i \
#       #   --seq_len $seq_len \
#       #   --gpu $gpu --num_train $num_train --batch_size 64

#       python scripts/sample_pl_new_c_diffstep_ablation.py -dc $j \
#         --model_name $mc \
#         --num_train $num_train \
#         --save_dir $save_dir \
#         --w_cond 0 \
#         --n_sample $n_sample \
#         --deterministic \
#         --gpu $gpu \
#         --seq_len $seq_len \
#         --pred_len $i --fast_sample
#     done
#   done
# done
