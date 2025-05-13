import torch
from torch import nn
# from .fourier import complex_freq_to_real_imag, real_imag_to_complex_freq


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
    f.append(n)
    f = [1] + f
    return f


# downsampling through moving average
class MovingAvgTime(nn.Module):
    """
    Moving average block to highlight the trend of time series, only for factors kernal size
    """

    def __init__(self, kernel_size, seq_length: int, stride=1):
        super(MovingAvgTime, self).__init__()
        self.kernel_size = kernel_size
        self.seq_length = seq_length
        K = torch.zeros(seq_length, int((seq_length - kernel_size) / stride + 1))
        start = 0
        for i in range(K.shape[1]):
            end = start + kernel_size
            K[start:end, i] = 1 / kernel_size
            start += stride
        K = K.unsqueeze(0)
        mode = "nearest-exact" if stride == 1 else "linear"
        self.K = (
            torch.nn.functional.interpolate(K, size=seq_length, mode=mode).squeeze().T
        )

        # self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor):
        # print(x.shape)
        assert x.shape[1] == self.seq_length
        
        x = self.K.to(x.device) @ x
        return x


# # downsampling through moving average
# class MovingAvgFreq(torch.nn.Module):
#     """
#     Moving average block to highlight the trend of time series, only for factors kernal size
#     """

#     def __init__(
#         self,
#         kernel_size: int,
#         seq_length: int,
#         sample_rate: float = 1.0,
#         # real_imag: bool = True,
#     ):
#         super().__init__()
#         self.seq_len = seq_length
#         freq = torch.pi * torch.linspace(0, seq_length - 1, seq_length) / seq_length
#         # freq = torch.fft.fftfreq(seq_length)
#         omega = 2 * torch.pi * freq / sample_rate
#         coeff = torch.exp(-1j * omega * (kernel_size - 1) / 2) / kernel_size
#         omega = torch.where(omega == 0, 1e-5, omega)

#         K = coeff * torch.sin(omega * kernel_size / 2) / torch.sin(omega / 2)
#         self.K = torch.diag(K)
#         # self.
#         # K =
#         # K.reshape(1, -1, 1)
#         # self.real_imag = real_imag

#     def forward(self, x: torch.Tensor):
#         """_summary_

#         Args:
#             x (torch.Tensor): complex tensor

#         Returns:
#             torch.Tensor: complex tensor
#         """
#         if torch.is_complex(x):
#             # print('directly multi')
#             x_filtered = self.K.to(x.device) @ x
#         else:
#             # print('first2complex')
#             x_filtered = self.K.to(x.device) @ real_imag_to_complex_freq(x)
#             x_filtered = complex_freq_to_real_imag(x_filtered, self.seq_len)
#         return x_filtered
