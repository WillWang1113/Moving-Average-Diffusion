import torch
import torch.nn as nn
from torch.nn import functional as F
from ..layers.Invertible import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name

        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len)
        )
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        # self.dropout = nn.Dropout(configs.dropout)
        self.rev = RevIN(configs.enc_in) if configs.rev else None
        self.pred_len = configs.pred_len
        self.channel = configs.enc_in 
        self.linear = nn.Linear(self.pred_len, 1)
    # def forward_loss(self, pred, true):
    #     return F.mse_loss(pred, true)

    def forecast(self, x_enc):
        # x: [B, L, D]
        x_enc = self.rev(x_enc, 'norm') if self.rev else x_enc
        x_enc = x_enc + self.temporal(x_enc.transpose(1, 2)).transpose(1, 2)
        pred = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm') if self.rev else pred
        pred = self.linear(pred.permute(0,2,1)).permute(0,2,1)
        # print(pred.shape)
        return pred
    
    def forward(self, observed_data, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(observed_data)
            return dec_out  # [B, L, D]
        else:
            raise ValueError('only forecast')
        return None