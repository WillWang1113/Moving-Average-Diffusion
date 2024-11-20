#!/bin/bash

gpu=0
model_config=(MADT MADT_FO_SETKS MADT_FO_SETKS_FD)
num_train=5
save_dir=/home/user/data/MAD_sr/savings
# kernel_size=(32)
kernel_size=(2 4 8 16 32)
pred_len=720


##############################################
##################### SR #####################
##############################################

# # # Train
# for mc in "${model_config[@]}"; do
#   python -u scripts/train_pl_new.py \
#     -dc exchange_rate \
#     -mc $mc \
#     --save_dir $save_dir \
#     --pred_len $pred_len \
#     --gpu $gpu --num_train $num_train --batch_size 64

#   for ks in "${kernel_size[@]}"; do
#     echo $ks $mc
#     python scripts/sample_pl_new.py -dc exchange_rate \
#       --model_name "${mc}_bs64" \
#       --num_train $num_train \
#       --save_dir $save_dir \
#       --condition sr \
#       --w_cond 0 \
#       --pred_len $pred_len \
#       --n_sample 100 \
#       --deterministic \
#       --gpu $gpu --kernel_size $ks --start_ks $ks

#   done
# done

############### DDPM ############
# # Train
for ks in "${kernel_size[@]}"; do

  python -u scripts/train_pl_new.py \
    -dc exchange_rate \
    -mc DDPM \
    --save_dir $save_dir \
    --pred_len $pred_len \
    --gpu $gpu --num_train $num_train --batch_size 64 \
    --condition sr --kernel_size $ks

  # python scripts/sample_pl_new.py -dc exchange_rate \
  #   --model_name "DDPM_bs64" \
  #   --num_train $num_train \
  #   --save_dir $save_dir \
  #   --condition sr \
  #   --w_cond 0 \
  #   --pred_len $pred_len \
  #   --n_sample 1 \
  #   --deterministic \
  #   --gpu $gpu --kernel_size $ks --start_ks $ks

done

##############  MFRED  #################

save_dir=/home/user/data/MAD_sr/savings
kernel_size=(2 4 8 16 32)
pred_len=576

# # # Train
# for mc in "${model_config[@]}"; do
#   for ks in "${kernel_size[@]}"; do
#     echo $ks $mc

#     python -u scripts/train_pl_new.py \
#       -dc mfred \
#       -mc $mc \
#       --save_dir $save_dir \
#       --pred_len $pred_len \
#       --gpu $gpu --num_train $num_train --batch_size 64 

#     python scripts/sample_pl_new.py -dc mfred \
#       --model_name "${mc}_bs64" \
#       --num_train $num_train \
#       --save_dir $save_dir \
#       --condition sr \
#       --w_cond 0 \
#       --pred_len $pred_len \
#       --n_sample 100 \
#       --deterministic \
#       --gpu $gpu --kernel_size $ks --start_ks $ks

#   done
# done

############### DDPM ############
# # Train
for ks in "${kernel_size[@]}"; do

  python -u scripts/train_pl_new.py \
    -dc mfred \
    -mc DDPM \
    --save_dir $save_dir \
    --pred_len $pred_len \
    --gpu $gpu --num_train $num_train --batch_size 64 \
    --condition sr --kernel_size $ks

  # python scripts/sample_pl_new.py -dc mfred \
  #   --model_name "DDPM_bs64" \
  #   --num_train $num_train \
  #   --save_dir $save_dir \
  #   --condition sr \
  #   --w_cond 0 \
  #   --pred_len $pred_len \
  #   --n_sample 100 \
  #   --deterministic \
  #   --gpu $gpu --kernel_size $ks --start_ks $ks

done
