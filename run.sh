#!/bin/bash


python scripts/train.py -c configs/MAD.yaml
python scripts/train.py -c configs/MAD.yaml --diff_config.norm
python scripts/train.py -c configs/MAD.yaml --diff_config.pred_diff
python scripts/train.py -c configs/MAD.yaml --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -c configs/MAD.yaml --diff_config.name MADFreq
# python scripts/train.py -c configs/MAD.yaml --diff_config.name MADFreq --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --diff_config.name MADFreq --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff

python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 
python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm
python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.pred_diff
python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff

python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine
python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine --diff_config.norm
python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine --diff_config.pred_diff
python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine --diff_config.name MADFreq
# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine --diff_config.name MADFreq --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine --diff_config.name MADFreq --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff

python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 
python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm
python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.pred_diff
python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq
# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule cosine --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff

python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm
python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm --diff_config.norm
python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm --diff_config.pred_diff
python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm --diff_config.name MADFreq
# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm --diff_config.name MADFreq --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm --diff_config.name MADFreq --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff

python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 
python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm
python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.pred_diff
python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq
# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --diff_config.noise_schedule ddpm --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff





# python scripts/sample.py --model_name MADFreq --deterministic --fast_sample --collect_all --kind freq --n_sample 100
python scripts/sample.py --model_name MADTime --deterministic --fast_sample --collect_all --kind time --n_sample 100