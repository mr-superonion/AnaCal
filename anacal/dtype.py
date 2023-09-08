import numpy as np
from dataclasses import dataclass, field


# Methods


@dataclass(frozen=True)
class FPFSMethod:
    sigma_as: float
    sigma_det: float
    rcut: int = field(default=32)
    nnord: int = field(default=4)
    ratio: float = field(default=1.6)
    c0: float = field(default=6.0)
    c2: float = field(default=20.0)
    alpha: float = field(default=1.0)
    beta: float = field(default=0.0)
    snr_min: float = field(default=10.0)
    noise_rev: bool = True
    method: str = field(default="fpfs", init=False)
    dtype: str = field(default="method", init=False)


# Data
# Image


@dataclass(frozen=True)
class ImageData:
    image: np.ndarray
    psf: np.ndarray = field(repr=False)
    noise_pow: np.ndarray = field(repr=False)
    scale: float
    mag_zero: float
    dtype: str = field(default="image", init=False)


# Catalog


@dataclass(frozen=True)
class FPFSCatalog:
    catalog: np.ndarray = field(repr=False)
    position: np.ndarray = field(repr=False)
    cov_mat: np.ndarray = field(repr=False)
    rcut: int = field(default=32)
    nnord: int = field(default=4)
    method: str = field(default="fpfs", init=False)
    dtype: str = field(default="catalog", init=False)
