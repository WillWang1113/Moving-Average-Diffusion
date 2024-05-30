#!/bin/bash

python scripts/train.py -c configs/MAD.yaml
python scripts/train.py -c configs/MAD.yaml --diff_config.norm
python scripts/train.py -c configs/MAD.yaml --diff_config.pred_diff
python scripts/train.py -c configs/MAD.yaml --diff_config.norm --diff_config.pred_diff

python scripts/train.py -c configs/MAD.yaml --diff_config.name MATFreq
python scripts/train.py -c configs/MAD.yaml --diff_config.name MATFreq --diff_config.norm
python scripts/train.py -c configs/MAD.yaml --diff_config.name MATFreq --diff_config.pred_diff
python scripts/train.py -c configs/MAD.yaml --diff_config.name MATFreq --diff_config.norm --diff_config.pred_diff

python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 
python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm
python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.pred_diff
python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm --diff_config.pred_diff

python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MATFreq
python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MATFreq --diff_config.norm
python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MATFreq --diff_config.pred_diff
python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MATFreq --diff_config.norm --diff_config.pred_diff


# python scripts/train.py -c configs/MAD.yaml
# python scripts/train.py -c configs/MAD.yaml --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -c configs/MAD.yaml --diff_config.name MATFreq
# python scripts/train.py -c configs/MAD.yaml --diff_config.name MATFreq --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --diff_config.name MATFreq --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --diff_config.name MATFreq --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.norm --diff_config.pred_diff

# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MATFreq
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MATFreq --diff_config.norm
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MATFreq --diff_config.pred_diff
# python scripts/train.py -c configs/MAD.yaml --bb_config.name ResNetBackbone --bb_config.hidden_size 128 --diff_config.name MATFreq --diff_config.norm --diff_config.pred_diff
