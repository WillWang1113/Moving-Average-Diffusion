#!/bin/bash

gpu=0
model_config=(MATSD_sr)
# model_config=(MADTC_NFD_DIT_eps_T25 MADTC_NFD_DIT_eps_T50 MADTC_NFD_DIT_eps_T75 MADTC_NFD_DIT_eps_T100 MADTC_NFD_DIT_eps_T150 MADTC_NFD_DIT_eps_T200)

num_train=5
save_dir=savings/sr
# kernel_size=(3)
kernel_size=(3 6 12)
pred_len=576
data_pth=(mfred wind solar)
n_sample=10

#############################################
#################### SR #####################
#############################################

# Train
for mc in "${model_config[@]}"; do
  for dataset in "${data_pth[@]}"; do

    python -u scripts/train_sr.py \
      -dc $dataset \
      -mc $mc \
      --save_dir $save_dir \
      --pred_len $pred_len \
      --gpu $gpu --num_train $num_train --batch_size 64

    for ks in "${kernel_size[@]}"; do
      echo $ks $mc
      python scripts/sample_sr.py -dc $dataset \
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
done


# Benchmark
# gpu=1
model_config=(DDPM)
# pred_len=576
# num_train=1
# save_dir=savings/sr
# save_dir=/home/user/data/MAD_sr_benchcos/savings
# kernel_size=(3)
# kernel_size=(3 6 12)
# n_sample=10
# data_pth=(wind solar mfred)

for mc in "${model_config[@]}"; do
  for dataset in "${data_pth[@]}"; do

    for ks in "${kernel_size[@]}"; do

      python -u scripts/train_sr.py \
        -dc $dataset \
        -mc $mc \
        --save_dir $save_dir \
        --pred_len $pred_len \
        --gpu $gpu --num_train $num_train --batch_size 64 \
        --condition sr --kernel_size $ks

      python scripts/sample_sr.py -dc $dataset \
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

