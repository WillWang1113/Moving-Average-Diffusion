# (ICML2025) A Non-isotropic Time Series Diffusion Model with Moving Average Transitions
This repo is the Pytorch implementation of our ICML'25 paper:

- Chenxi Wang, Linxiao Yang, Zhixian Wang, Liang Sun, Yi Wang*, "A Non-isotropic Time Series Diffusion Model with Moving Average Transitions," in Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada, 2025

## Requirements
Python version: 3.10

The must-have packages can be installed by running
```
pip install requirements.txt
python setup.py develop
```

## How to run
Experiments include time series forecasting (fcst), super-resolution (sr) and synthesis (uncond). Run the bash script:
```
bash run_{fcst, sr, uncond}.sh
```

Hyperparameters can be adjusted in `configs/`, and more arguments about training/sampling, see `scripts/{train, sample}_{fcst, sr, uncond}.py`



