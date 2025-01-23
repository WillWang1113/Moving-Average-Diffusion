#!/bin/bash

gpu=1
model_config=(DDPM_T100)
pred_len=576
num_train=5
save_dir=/home/user/data/MAD_sr_benchcos/savings
kernel_size=(3 6 12)
n_sample=10
data_pth=(solar)

for mc in "${model_config[@]}"; do
  for dataset in "${data_pth[@]}"; do

    for ks in "${kernel_size[@]}"; do

      python -u scripts/train_pl_sr.py \
        -dc $dataset \
        -mc $mc \
        --save_dir $save_dir \
        --pred_len $pred_len \
        --gpu $gpu --num_train $num_train --batch_size 64 \
        --condition sr --kernel_size $ks

      # python scripts/sample_pl_new_c_sr.py -dc $dataset \
      #   --model_name "${mc}_bs64_condsr_ks${ks}" \
      #   --num_train $num_train \
      #   --save_dir $save_dir \
      #   --condition sr \
      #   --w_cond 0 \
      #   --pred_len $pred_len \
      #   --n_sample $n_sample \
      #   --deterministic \
      #   --gpu $gpu --kernel_size $ks --start_ks $ks --strategy ddpm
    done
  done
done




# SAMPLE
gpu=1
model_config=(DDPM_T25 DDPM_T50 DDPM_T75 DDPM_T100)
pred_len=576
num_train=5
save_dir=/home/user/data/MAD_sr_benchcos/savings
kernel_size=(3 6 12)
n_sample=10
data_pth=(mfred wind solar)

for mc in "${model_config[@]}"; do
  for dataset in "${data_pth[@]}"; do

    for ks in "${kernel_size[@]}"; do

      # python -u scripts/train_pl_sr.py \
      #   -dc $dataset \
      #   -mc $mc \
      #   --save_dir $save_dir \
      #   --pred_len $pred_len \
      #   --gpu $gpu --num_train $num_train --batch_size 64 \
      #   --condition sr --kernel_size $ks

      python scripts/sample_pl_new_c_sr.py -dc $dataset \
        --model_name "${mc}_bs64_condsr_ks${ks}" \
        --num_train $num_train \
        --save_dir $save_dir \
        --condition sr \
        --w_cond 0 \
        --pred_len $pred_len \
        --n_sample $n_sample \
        --deterministic \
        --gpu $gpu --kernel_size $ks --start_ks $ks --strategy ddpm
    done
  done
done
