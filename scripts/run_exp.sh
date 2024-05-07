#!/bin/bash
rm -rf runs/

python run.py -s cnn_freq
python run.py -s cnn_time
python run.py -s mlp_time
python run.py -s mlp_freq
python run.py -s cmlp_freq


