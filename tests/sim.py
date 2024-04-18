import numpy as np
from scipy.special import erf


def tophat(x):
    x = np.asarray(x)
    y = np.zeros_like(x)
    msk = (x > -np.pi) & (x < np.pi)
    y[msk] = 1.0
    return y


def gaussian_top_hat_convolution_1D(d, sigma, values):
    return 0.5 * (
        erf((values + d) / (np.sqrt(2) * sigma))
        - erf((values - d) / (np.sqrt(2) * sigma))
    )


def gaussian_top_hat_convolution_2D(d, sigma, x, y):
    gx = gaussian_top_hat_convolution_1D(d, sigma, x)
    gy = gaussian_top_hat_convolution_1D(d, sigma, y)
    return gx * gy


def apply_transformations(x, y, theta, gamma1, gamma2):
    rotation_matrix = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    shear_matrix = np.array([[1 - gamma1, -gamma2], [-gamma2, 1 + gamma1]])
    tm = np.dot(rotation_matrix, shear_matrix)

    x_prime = tm[0, 0] * x + tm[0, 1] * y
    y_prime = tm[1, 0] * x + tm[1, 1] * y
    return x_prime, y_prime


def gauss_kernel_rfft(
    ny: int,
    nx: int,
    sigma: float,
    klim: float,
    theta: float,
    gamma1: float,
    gamma2: float,
):
    """Generates a Gaussian kernel on grids for np.fft.rfft transform
    The kernel is truncated at radius klim.

    Args:
    ny (int):    		    grid size in y-direction
    nx (int):    		    grid size in x-direction
    sigma (float):		    scale of Gaussian in Fourier space (pixel scale=1)
    klim (float):           upper limit of k
    theta (float):          rotation angle
    gamma1 (float):         shear, the first component
    gamma2 (float):         shear, the second component

    Returns:
    out (ndarray):          Gaussian on grids
    """
    x = np.fft.rfftfreq(nx, 1 / np.pi / 2.0)
    y = np.fft.fftfreq(ny, 1 / np.pi / 2.0)
    ygrid, xgrid = np.meshgrid(y, x, indexing="ij")
    x2, y2 = apply_transformations(xgrid, ygrid, theta, gamma1, gamma2)
    r2 = x2**2.0 + y2**2.0
    mask = (r2 <= klim**2).astype(int)
    out = np.exp(-r2 / 2.0 / sigma**2.0) * mask
    return out


def gauss_tophat_kernel_rfft(
    ny: int,
    nx: int,
    d: float,
    sigma: float,
    theta: float,
    gamma1: float,
    gamma2: float,
):
    """
    Args:
    ny (int):    		    grid size in y-direction
    nx (int):    		    grid size in x-direction
    d (float):              width / 2 of top hat function
    sigma (float):		    scale of Gaussian in Fourier space (pixel scale=1)
    theta (float):          rotation angle
    gamma1 (float):         shear, the first component
    gamma2 (float):         shear, the second component

    Returns:
    out (ndarray):          Gaussian x Tophat on grids
    """
    x = np.fft.rfftfreq(nx, 1 / np.pi / 2.0)
    y = np.fft.fftfreq(ny, 1 / np.pi / 2.0)
    ygrid, xgrid = np.meshgrid(y, x, indexing="ij")
    x2, y2 = apply_transformations(xgrid, ygrid, theta, gamma1, gamma2)
    out = gaussian_top_hat_convolution_2D(d, sigma, x2, y2)
    return out
