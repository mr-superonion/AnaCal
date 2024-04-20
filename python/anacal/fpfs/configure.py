from pydantic import BaseModel


class FpfsConfig(BaseModel):
    rcut: int = 32
    psf_rcut: int = 26
    noise_rev: bool = False
    nord: int = 4
    det_nrot: int = 4
    klim_thres: float = 1e-12
    bound: int = 35
    sigma_arcsec: float = 0.52
    pratio: float = 0.00
    pthres: float = 0.8
    pthres2: float = 0.12
    snr_min: float = 12
    r2_min: float = 0.1
    c0: float = 5.0
    c2: float = 22.0
    alpha: float = 1.0
    beta: float = 0.0
