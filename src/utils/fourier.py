import torch
import math
from torch.fft import rfft, irfft

def dft(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Compute the DFT of the input time series by keeping only the non-redundant components.

    Args:
        x (torch.Tensor): Time series of shape (batch_size, max_len, n_channels).

    Returns:
        torch.Tensor: DFT of x with the same size (batch_size, max_len, n_channels).
    """

    max_len = x.size(1)

    # Compute the FFT until the Nyquist frequency
    dft_full = rfft(x, dim=1, norm="ortho")
    if kwargs.get("real_imag"):
        # Concatenate real and imaginary parts
        x_tilde = complex_freq_to_real_imag(dft_full)
        # assert (
        #     x_tilde.size() == x.size()
        # ), f"The DFT and the input should have the same size. Got {x_tilde.size()} and {x.size()} instead."
    else:
        x_tilde = dft_full
    return x_tilde.detach()


def idft(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Compute the inverse DFT of the input DFT that only contains non-redundant components.

    Args:
        x (torch.Tensor): DFT of shape (batch_size, max_len, n_channels).

    Returns:
        torch.Tensor: Inverse DFT of x with the same size (batch_size, max_len, n_channels).
    """
    max_len = x.size(1)
    if kwargs.get("real_imag"):
        x_freq = real_imag_to_complex_freq(x)
    else:
        x_freq = x
        max_len = 2 * (max_len - 1)

    # Apply IFFT
    x_time = irfft(x_freq, n=max_len, dim=1, norm="ortho")

    # assert isinstance(x_time, torch.Tensor)
    # assert (
    #     x_time.size() == x.size()
    # ), f"The inverse DFT and the input should have the same size. Got {x_time.size()} and {x.size()} instead."

    return x_time.detach()


def real_imag_to_complex_freq(x: torch.Tensor):
    max_len = x.size(1)
    n_real = math.ceil((max_len + 1) / 2)

    # Extract real and imaginary parts
    x_re = x[:, :n_real, :]
    x_im = x[:, n_real:, :]

    # Create imaginary tensor
    zero_padding = torch.zeros(size=(x.size(0), 1, x.size(2)), device=x.device)
    x_im = torch.cat((zero_padding, x_im), dim=1)

    # If number of time steps is even, put the null imaginary part
    if max_len % 2 == 0:
        x_im = torch.cat((x_im, zero_padding), dim=1)

    # assert (
    #     x_im.size() == x_re.size()
    # ), f"The real and imaginary parts should have the same shape, got {x_re.size()} and {x_im.size()} instead."

    x_freq = torch.complex(x_re, x_im)
    return x_freq


def complex_freq_to_real_imag(dft_full: torch.Tensor):
    max_len = 2 * (dft_full.shape[1] - 1)

    # Compute the FFT until the Nyquist frequency
    dft_re = torch.real(dft_full)
    dft_im = torch.imag(dft_full)

    # The first harmonic corresponds to the mean, which is always real
    # zero_padding = torch.zeros_like(dft_im[:, 0, :], device=dft_full.device)
    # assert torch.allclose(
    #     dft_im[:, 0, :], zero_padding
    # ), f"The first harmonic of a real time series should be real, yet got imaginary part {dft_im[:, 0, :]}."
    dft_im = dft_im[:, 1:]

    # If max_len is even, the last component is always zero
    if max_len % 2 == 0:
        # assert torch.allclose(
        #     dft_im[:, -1, :], zero_padding
        # ), f"Got an even {max_len=}, which should be real at the Nyquist frequency, yet got imaginary part {dft_im[:, -1, :]}."
        dft_im = dft_im[:, :-1]

    # Concatenate real and imaginary parts
    x_tilde = torch.cat((dft_re, dft_im), dim=1)
    return x_tilde


def sphere2complex(theta: torch.Tensor, phi: torch.Tensor):
    r"""Spherical Riemann stereographic projection coordinates to complex number coordinates. I.e. inverse Spherical Riemann stereographic projection map.

    The inverse transform, :math:`v: \mathcal{D} \rightarrow \mathbb{C}`, is given as

    .. math::
        \begin{aligned}
            s = v(\theta, \phi) = \tan \left( \frac{\phi}{2} + \frac{\pi}{4} \right) e^{i \theta}
        \end{aligned}

    Args:
        theta (Tensor): Spherical Riemann stereographic projection coordinates, shape :math:`\theta` component of shape :math:`(\text{dimension})`.
        phi (Tensor): Spherical Riemann stereographic projection coordinates, shape :math:`\phi` component of shape :math:`(\text{dimension})`.

    Returns:
        Tuple Tensor of real and imaginary components of the complex numbers coordinate, of :math:`(\Re(s), \Im(s))`. Where :math:`s \in \mathbb{C}^d`, with :math:`d` is :math:`\text{dimension}`.

    """
    # dim(phi) == dim(theta), takes in spherical co-ordinates returns comlex real & imaginary parts
    if phi.shape != theta.shape:
        raise ValueError("Invalid phi theta shapes")
    r = torch.tan(phi / 2 + torch.pi / 4)
    s_real = r * torch.cos(theta)
    s_imag = r * torch.sin(theta)

    # s = torch.complex(s_real, s_imag)
    assert s_real.shape == theta.shape
    # return s

    return s_real, s_imag


def complex2sphere(s_real: torch.Tensor, s_imag: torch.Tensor):
    r"""Complex coordinates to to Spherical Riemann stereographic projection coordinates.
    I.e. we can translate any complex number :math:`s\in \mathbb{C}` into a coordinate on the Riemann Sphere :math:`(\theta, \phi) \in \mathcal{D} = (-{\pi}, {\pi}) \times (-\frac{\pi}{2}, \frac{\pi}{2})`, i.e.

    .. math::
        \begin{aligned}
            u(s) = \left( \arctan \left( \frac{\Im(s)}{\Re(s)} \right),\arcsin \left( \frac{|s|^2-1}{|s|^2+1} \right) \right)
        \end{aligned}

    For more details see `[1] <https://arxiv.org/abs/2206.04843>`__.

    Args:
        s_real (Tensor): Real component of the complex tensor, of shape :math:`(\text{dimension})`.
        s_imag (Tensor): Imaginary component of the complex tensor, of shape :math:`(\text{dimension})`.

    Returns:
        Tuple Tensor of :math:`(\theta, \phi)` of complex number in spherical Riemann stereographic projection coordinates. Where the shape  of :math:`\theta, \phi` is of shape :math:`(\text{dimension})`.

    """

    # din(s_real) == dim(s_imag), takes in real & complex parts returns spherical co-ordinates
    if s_real.shape != s_imag.shape:
        s_real = s_real.expand_as(s_imag)
    assert s_real.shape == s_imag.shape
    s_abs_2 = s_imag**2 + s_real**2
    # Handle points at infinity
    phi_r_int = torch.where(
        torch.isinf(s_abs_2), torch.ones_like(s_abs_2), ((s_abs_2 - 1) / (s_abs_2 + 1))
    )
    phi_r = torch.asin(phi_r_int)
    theta_r = torch.atan2(s_imag, s_real)
    return theta_r, phi_r


def moving_average_freq_response(N: int, sample_rate: float, freq: torch.Tensor):
    omega = 2 * torch.pi * freq / sample_rate
    coeff = torch.exp(-1j * omega * (N - 1) / 2) / N
    omega = torch.where(omega == 0, 1e-5, omega)
    Hw = coeff * torch.sin(omega * N / 2) / torch.sin(omega / 2)
    return Hw



def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    #Vc = torch.fft.rfft(v, 1)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = - torch.arange(N, dtype=x.dtype,
                       device=x.device)[None, :] * torch.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= torch.sqrt(torch.tensor(N)) * 2
        V[:, 1:] /= torch.sqrt(torch.tensor(N) / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= torch.sqrt(torch.tensor(N)) * 2
        X_v[:, 1:] *= torch.sqrt(torch.tensor(N) / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype,
                     device=X.device)[None, :] * torch.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    #v = torch.fft.irfft(V, 1)
    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)
