import abc
from typing import Dict
import torch
from torch import nn


class BaseModel(abc.ABC, nn.Module):
    def train_step(self, batch):
        raise NotImplementedError()

    def validation_step(self, batch):
        raise NotImplementedError()

    def predict_step(self, batch):
        raise NotImplementedError()


class BaseDiffusion(BaseModel):
    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError()

    def configure_sampling(self, *args, **kwargs):
        raise NotImplementedError()
    
    # def init_noise(self, *args, **kwargs):
    #     raise NotImplementedError()

    # def _encode_condition(self, *args, **kwargs):
    #     raise NotImplementedError()
    
    
