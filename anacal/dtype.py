import numpy as np
from dataclasses import dataclass, field


# Methods

@dataclass(frozen=True)
class FPFSMethod:
    sigma_as: float
    sigma_det: float
    rcut: int = field(default=32)
    nnord: int = field(default=4)
    method: str = field(default="fpfs", init=False)
    dtype: str = field(default="method", init=False)


# Data
# Image

@dataclass(frozen=True)
class ImageData:
    image: np.ndarray
    psf: np.ndarray
    noise_pow: np.ndarray
    scale: float
    mag_zero: float
    dtype: str = field(default="image", init=False)


# Catalog

@dataclass(frozen=True)
class FPFSCatalog:
    catalog: np.ndarray
    position: np.ndarray
    rcut: int = field(default=32)
    nnord: int = field(default=4)
    method: str = field(default="fpfs", init=False)
    dtype: str = field(default="catalog", init=False)
