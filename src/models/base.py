import abc
from typing import Dict
import torch
from torch import nn



class BaseDiffusion(abc.ABC):
    
    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError()

    def configure_sampling(self, *args, **kwargs):
        raise NotImplementedError()
    
    
    
