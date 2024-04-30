#!/bin/bash
rm -rf runs/

# python run.py -d mfred -s mfred_cnn_freq
# python run.py -d mfred -s mfred_cnn_time
python run.py -d mfred -s mfred_freqlinear_time
python run.py -d mfred -s mfred_mlp_freq
python run.py -d mfred -s mfred_mlp_time


