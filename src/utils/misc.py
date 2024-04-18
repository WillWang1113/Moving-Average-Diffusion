import torch
from torch import nn

def get_factors(n):
    f = list(set(
        factor for i in range(2, int(n**0.5) + 1) if n % i == 0
        for factor in (i, n//i)
    ))
    f.sort()
    return f

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - self.kernel_size // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x