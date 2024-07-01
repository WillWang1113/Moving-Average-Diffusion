#!/bin/bash


# python scripts/train.py -mc MAD.yaml
# python scripts/train.py -mc MAD.yaml --diff_config.norm
# python scripts/train.py -mc MAD.yaml --diff_config.pred_diff
# python scripts/train.py -mc MAD.yaml --diff_config.norm --diff_config.pred_diff

# # python scripts/train.py -mc MAD.yaml --diff_config.name MADFreq
# # python scripts/train.py -mc MAD.yaml --diff_config.name MADFreq --diff_config.norm
# # python scripts/train.py -mc MAD.yaml --diff_config.name MADFreq --diff_config.pred_diff
# # python scripts/train.py -mc MAD.yaml --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -mc MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 
# python scripts/train.py -mc MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm
# python scripts/train.py -mc MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.pred_diff
# python scripts/train.py -mc MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm --diff_config.pred_diff

# # python scripts/train.py -mc MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq
# # python scripts/train.py -mc MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm
# # python scripts/train.py -mc MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.pred_diff
# # python scripts/train.py -mc MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine
# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine --diff_config.norm
# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine --diff_config.pred_diff
# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine --diff_config.norm --diff_config.pred_diff

# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine --diff_config.name MADFreq
# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine --diff_config.name MADFreq --diff_config.norm
# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine --diff_config.name MADFreq --diff_config.pred_diff
# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 
# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm
# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.pred_diff
# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm --diff_config.pred_diff

# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq
# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm
# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.pred_diff
# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm
# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm --diff_config.norm
# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm --diff_config.pred_diff
# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm --diff_config.norm --diff_config.pred_diff

# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm --diff_config.name MADFreq
# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm --diff_config.name MADFreq --diff_config.norm
# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm --diff_config.name MADFreq --diff_config.pred_diff
# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 
# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm
# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.pred_diff
# python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm --diff_config.pred_diff

# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq
# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm
# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.pred_diff
# # python scripts/train.py -mc MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff







python scripts/train.py -mc MAD --dc mfred  --n_out 144
python scripts/train.py -mc MAD --dc mfred --diff_config.pred_diff --n_out 144

python scripts/train.py -mc MAD --dc mfred --diff_config.name MADFreq  --n_out 144
python scripts/train.py -mc MAD --dc mfred --diff_config.name MADFreq --diff_config.pred_diff --n_out 144

python scripts/train.py -mc MAD --dc mfred --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --n_out 144
python scripts/train.py -mc MAD --dc mfred --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.pred_diff --n_out 144

python scripts/train.py -mc MAD --dc mfred --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --n_out 144
python scripts/train.py -mc MAD --dc mfred --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.pred_diff --n_out 144


python scripts/train.py -mc MAD --dc mfred --n_out 432
python scripts/train.py -mc MAD --dc mfred --diff_config.pred_diff --n_out 432

python scripts/train.py -mc MAD --dc mfred --diff_config.name MADFreq --n_out 432
python scripts/train.py -mc MAD --dc mfred --diff_config.name MADFreq --diff_config.pred_diff --n_out 432

python scripts/train.py -mc MAD --dc mfred --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --n_out 432
python scripts/train.py -mc MAD --dc mfred --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.pred_diff --n_out 432

python scripts/train.py -mc MAD --dc mfred --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --n_out 432
python scripts/train.py -mc MAD --dc mfred --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.pred_diff --n_out 432

python scripts/sample.py --model_name MADFreq --deterministic --fast_sample --collect_all --kind freq --n_sample 100
python scripts/sample.py --model_name MADTime --deterministic --fast_sample --collect_all --kind time --n_sample 100
