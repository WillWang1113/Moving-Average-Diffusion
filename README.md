# A Non-isotropic Time Series Diffusion Model with Moving Average Transitions
This repo is the Pytorch implementation of the paper: A Non-isotropic Time Series Diffusion Model with Moving Average Transitions

## Requirements
Python version: 3.10

The must-have packages can be installed by running
```
pip install requirements.txt
python setup.py develop
```

## How to run
Experiments include time series forecasting (fcst), super-resolution (sr) and model ablation (ablation). Run the bash script:
```
bash run_{fcst, sr, ablation}.sh
```

Hyperparameters can be adjusted in `configs/`, and more arguments about training/sampling, see `scripts/{train, sample}_{fcst, sr, ablation_module, ablation_diffstep}.py`



