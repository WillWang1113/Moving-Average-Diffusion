#!/bin/bash
#####################
# Backward strategy #
#####################

gpu=1
model_config=DDPM_IN_MA
ratio_diff_steps=(0.9 0.7 0.5 0.3 0.1 0.02 0.05)

save_dir=/home/user/data2/ICML_camera_ready_test/ablation
num_train=1
pred_len=576
seq_len=0
n_sample=5
data_pth=(
  wind
  mfred
  solar
)

# # # diffusion steps
# for j in "${data_pth[@]}"; do
#   python -u scripts/train_ablation.py \
#     -dc $j \
#     -mc $model_config \
#     --save_dir $save_dir \
#     --pred_len $pred_len \
#     --seq_len $seq_len \
#     --gpu $gpu --num_train $num_train --batch_size 64

#   for rds in "${ratio_diff_steps[@]}"; do

#     python scripts/sample_ablation_diffstep.py -dc $j \
#       --model_name $model_config \
#       --num_train $num_train \
#       --save_dir $save_dir \
#       --w_cond 0 \
#       --n_sample $n_sample \
#       --deterministic \
#       --gpu $gpu \
#       --seq_len $seq_len \
#       --pred_len $pred_len --num_diff_steps $rds

#   done
# done

#####################
# Diffusion Modules #
#####################
gpu=1
model_config=(DDPM DDPM_IN DDPM_MA DDPM_IN_MA)


# Modules
for j in "${data_pth[@]}"; do
  for mc in "${model_config[@]}"; do

      python -u scripts/train_ablation.py \
        -dc $j \
        -mc $mc \
        --save_dir $save_dir \
        --pred_len $pred_len \
        --seq_len $seq_len \
        --gpu $gpu --num_train $num_train --batch_size 64

      python scripts/sample_ablation_module.py -dc $j \
        --model_name $mc \
        --num_train $num_train \
        --save_dir $save_dir \
        --w_cond 0 \
        --n_sample $n_sample \
        --deterministic \
        --gpu $gpu \
        --seq_len $seq_len \
        --pred_len $pred_len

    done
  done
done
