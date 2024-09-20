import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Invertible import RevIN



class Model(nn.Module):
    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.Linear = nn.ModuleList([
            nn.Linear(configs.seq_len, configs.pred_len) for _ in range(configs.channel)
        ]) if individual else nn.Linear(configs.seq_len, configs.pred_len)
        
        self.dropout = nn.Dropout(configs.dropout)
        self.rev = RevIN(configs.enc_in) if configs.rev else None
        self.individual = individual
        self.pred_len = configs.pred_len
        self.channel = configs.enc_in
        self.linear = nn.Linear(self.pred_len, 1)

    # def forward_loss(self, pred, true):
    #     return F.mse_loss(pred, true)

    def forecast(self, x_enc):
        # x: [B, L, D]
        x_enc = self.rev(x_enc, 'norm') if self.rev else x_enc
        x_enc = self.dropout(x_enc)
        if self.individual:
            pred = torch.zeros((x_enc.shape[0], self.pred_len, self.channel))
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x_enc[:, :, idx])
        else:
            pred = self.Linear(x_enc.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm') if self.rev else pred
        pred = self.linear(pred.permute(0,2,1)).permute(0,2,1)
        return pred
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError('only forecast')
        return None