#!/bin/bash

python run.py -d mfred -s mfred_cnn_freq
python run.py -d mfred -s mfred_cnn_time
python run.py -d mfred -s mfred_mlp_freq
python run.py -d mfred -s mfred_mlp_time

python run.py -d nrel -s nrel_cnn_freq
python run.py -d nrel -s nrel_cnn_time
python run.py -d nrel -s nrel_mlp_freq
python run.py -d nrel -s nrel_mlp_time