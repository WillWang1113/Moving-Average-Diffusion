import torch
import math
from torch.fft import rfft, irfft


def dft(x: torch.Tensor, stereographic=False) -> torch.Tensor:
    """Compute the DFT of the input time series by keeping only the non-redundant components.

    Args:
        x (torch.Tensor): Time series of shape (batch_size, max_len, n_channels).

    Returns:
        torch.Tensor: DFT of x with the same size (batch_size, max_len, n_channels).
    """

    # Compute the FFT until the Nyquist frequency
    dft_full = rfft(x, dim=1, norm="ortho")
    dft_re = torch.real(dft_full)
    dft_im = torch.imag(dft_full)

    if stereographic:
        # stereographic projection
        theta, phi = complex2sphere(dft_re, dft_im)
        x_tilde = torch.cat((theta, phi), dim=1)
    else:
        x_tilde = torch.cat((dft_re, dft_im), dim=1)
    return x_tilde.detach()


def idft(x: torch.Tensor, stereographic=False) -> torch.Tensor:
    """Compute the inverse DFT of the input DFT that only contains non-redundant components.

    Args:
        x (torch.Tensor): DFT of shape (batch_size, max_len, n_channels).

    Returns:
        torch.Tensor: Inverse DFT of x with the same size (batch_size, max_len, n_channels).
    """
    max_len = x.size(1)
    n_real = int(0.5 * max_len)
    if stereographic:
        theta = x[:, :n_real, :]
        phi = x[:, n_real:, :]
        x_re, x_im = sphere2complex(theta, phi)
        x_freq = torch.complex(x_re, x_im)
        x_time = irfft(x_freq, dim=1, norm="ortho")
    else:
        # Extract real and imaginary parts
        x_re = x[:, :n_real, :]
        x_im = x[:, n_real:, :]

        x_freq = torch.complex(x_re, x_im)

        # Apply IFFT
        x_time = irfft(x_freq, dim=1, norm="ortho")

    assert isinstance(x_time, torch.Tensor)

    return x_time.detach()


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
