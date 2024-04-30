import torch
from torch import nn

from src.utils.fourier import complex_freq_to_real_imag, real_imag_to_complex_freq


def get_factors(n):
    f = list(
        set(
            factor
            for i in range(2, int(n**0.5) + 1)
            if n % i == 0
            for factor in (i, n // i)
        )
    )
    f.sort()
    return f


# From Autoformer
# class MovingAvg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """

#     def __init__(self, kernel_size, stride=1):
#         super(MovingAvg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, self.kernel_size // 2, 1)
#         end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - self.kernel_size // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)
#         return x


# downsampling through moving average
class MovingAvgTime(nn.Module):
    """
    Moving average block to highlight the trend of time series, only for factors kernal size
    """

    def __init__(self, kernel_size, stride=1):
        super(MovingAvgTime, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor):
        orig_size = x.shape[1]

        # front = x[:, 0:1, :].repeat(1, self.kernel_size // 2, 1)
        # end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - self.kernel_size // 2, 1)
        # x = torch.cat([front, x, end], dim=1)

        x = self.avg(x.permute(0, 2, 1))
        x = nn.functional.interpolate(x, size=orig_size, mode="linear")
        x = x.permute(0, 2, 1)
        return x


# downsampling through moving average
class MovingAvgFreq(torch.nn.Module):
    """
    Moving average block to highlight the trend of time series, only for factors kernal size
    """

    def __init__(self, kernel_size: int, freq: torch.Tensor, sample_rate: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        omega = 2 * torch.pi * freq / sample_rate
        coeff = torch.exp(-1j * omega * (kernel_size - 1) / 2) / kernel_size
        omega = torch.where(omega == 0, 1e-5, omega)
        Hw = coeff * torch.sin(omega * kernel_size / 2) / torch.sin(omega / 2)
        self.Hw = Hw.reshape(1, -1, 1)

    def forward(self, x: torch.Tensor):
        x_complex = real_imag_to_complex_freq(x)
        x_filtered = x_complex * self.Hw.to(x.device)
        return complex_freq_to_real_imag(x_filtered, x.shape[1])
