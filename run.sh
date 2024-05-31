#!/bin/bash


# python scripts/train.py -c configs/MAD.yaml
# python scripts/train.py -c configs/MAD.yaml --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -c configs/MAD.yaml --diff_config.name MADFreq
# python scripts/train.py -c configs/MAD.yaml --diff_config.name MADFreq --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --diff_config.name MADFreq --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff


# python scripts/train.py -c configs/MAD.yaml
# python scripts/train.py -c configs/MAD.yaml --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -c configs/MAD.yaml --diff_config.name MADFreq
# python scripts/train.py -c configs/MAD.yaml --diff_config.name MADFreq --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --diff_config.name MADFreq --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MADFreq --diff_config.norm --diff_config.pred_diff



python scripts/sample.py --model_name MADFreq --deterministic --fast_sample --collect_all --kind freq
python scripts/sample.py --model_name MADTime --deterministic --fast_sample --collect_all --kind time