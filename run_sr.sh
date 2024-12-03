#!/bin/bash

gpu=1
# model_config=(MADTC_NFD_MLP_eps MADTC_NFD_DIT_eps)
model_config=(MADTC_NFD_DIT_eps)
# model_config=(MADTC_NFD_MLP_x0 MADTC_NFD_MLP_eps MADTC_NFD_DIT_eps MADTC_NFD_DIT_x0)

# model_config=(MADTC_NFD_MLP MADTC_FD_MLP)
num_train=5
save_dir=./savings/MAD_sr/savings
# save_dir=/home/user/data/MAD_sr/savings
# kernel_size=(32)
# kernel_size=(6)
kernel_size=(2 4 6 8 16)
pred_len=720
n_sample=1

##############################################
##################### SR #####################
##############################################

# Train
for mc in "${model_config[@]}"; do
  # python -u scripts/train_pl_sr.py \
  #   -dc exchange_rate \
  #   -mc $mc \
  #   --save_dir $save_dir \
  #   --pred_len $pred_len \
  #   --gpu $gpu --num_train $num_train --batch_size 64

  for ks in "${kernel_size[@]}"; do
    echo $ks $mc
    python scripts/sample_pl_new_c_sr.py -dc exchange_rate \
      --model_name "${mc}_bs64_condNone_ksNone" \
      --num_train $num_train \
      --save_dir $save_dir \
      --condition sr \
      --w_cond 0 \
      --pred_len $pred_len \
      --n_sample $n_sample \
      --deterministic \
      --gpu $gpu --kernel_size $ks --start_ks $ks

  done
done

# ############### DDPM ############
# # Train
for ks in "${kernel_size[@]}"; do

  # python -u scripts/train_pl_sr.py \
  #   -dc exchange_rate \
  #   -mc DDPM \
  #   --save_dir $save_dir \
  #   --pred_len $pred_len \
  #   --gpu $gpu --num_train $num_train --batch_size 64 \
  #   --condition sr --kernel_size $ks

  python scripts/sample_pl_new_c_sr.py -dc exchange_rate \
    --model_name "DDPM_bs64_condsr_ks${ks}" \
    --num_train $num_train \
    --save_dir $save_dir \
    --condition sr \
    --w_cond 0 \
    --pred_len $pred_len \
    --n_sample $n_sample \
    --deterministic \
    --gpu $gpu --kernel_size $ks --start_ks $ks

done

# ##############  MFRED  #################

# save_dir=/home/user/data/MAD_sr/savings
pred_len=576

# # Train
for mc in "${model_config[@]}"; do
  # python -u scripts/train_pl_sr.py \
  #   -dc mfred \
  #   -mc $mc \
  #   --save_dir $save_dir \
  #   --pred_len $pred_len \
  #   --gpu $gpu --num_train $num_train --batch_size 64

  for ks in "${kernel_size[@]}"; do
    echo $ks $mc
    python scripts/sample_pl_new_c_sr.py -dc mfred \
      --model_name "${mc}_bs64_condNone_ksNone" \
      --num_train $num_train \
      --save_dir $save_dir \
      --condition sr \
      --w_cond 0 \
      --pred_len $pred_len \
      --n_sample $n_sample \
      --deterministic \
      --gpu $gpu --kernel_size $ks --start_ks $ks

  done
done

############## DDPM ############
# Train
for ks in "${kernel_size[@]}"; do

  # python -u scripts/train_pl_sr.py \
  #   -dc mfred \
  #   -mc DDPM \
  #   --save_dir $save_dir \
  #   --pred_len $pred_len \
  #   --gpu $gpu --num_train $num_train --batch_size 64 \
  #   --condition sr --kernel_size $ks

  python scripts/sample_pl_new_c_sr.py -dc mfred \
    --model_name "DDPM_bs64_condsr_ks${ks}" \
    --num_train $num_train \
    --save_dir $save_dir \
    --condition sr \
    --w_cond 0 \
    --pred_len $pred_len \
    --n_sample $n_sample \
    --deterministic \
    --gpu $gpu --kernel_size $ks --start_ks $ks

done
