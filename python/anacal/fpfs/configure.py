from pydantic import BaseModel, Field


class FpfsConfig(BaseModel):
    rcut: int = Field(
        default=32,
        description="""Galaxies are put into stamp before measurement, rcut
            is the radius of the cut
        """,
    )
    psf_rcut: int = Field(
        default=26,
        description="""Cut off radius for PSF.
        """,
    )
    noise_rev: bool = Field(
        default=False,
        description="""Whether do noise bias correction. The noise bias is
            corrected by adding noise to image to evaluate noise reponse.
        """,
    )
    nord: int = Field(
        default=4,
        description="""Maximum radial number `n` to use for the shapelet basis
        """,
    )
    det_nrot: int = Field(
        default=4,
        description="""Number of directions to calculate when detecting the
            peaks.
        """,
    )
    klim_thres: float = Field(
        default=1e-12,
        description="""The threshold used to define the upper limit of k we use
        in Fourier space.
        """,
    )
    bound: int = Field(
        default=35,
        description="""Boundary buffer length, the sources in the buffer reion
        are not counted.
        """,
    )
    sigma_arcsec: float = Field(
        default=0.52,
        description="""Smoothing scale of the shapelet and detection kernel.
        """,
    )
    pratio: float = Field(
        default=0.00,
        description="""Detection parameter (peak identification) for the first
        pooling.
        """,
    )
    pthres: float = Field(
        default=0.8,
        description="""Detection threshold (peak identification) for the first
        pooling.
        """,
    )
    pthres2: float = Field(
        default=0.12,
        description="""Detection threshold (peak identification) for the second
        pooling.
        """,
    )
    fthres: float = Field(
        default=8.5,
        description="""Detection threshold (minimum signal-to-noise ratio) for
        the first pooling.
        """,
    )
    snr_min: float = Field(
        default=12,
        description="""Minimum Signal-to-Noise Ratio.
        """,
    )
    r2_min: float = Field(
        default=0.1,
        description="""Minimum resolution.
        """,
    )
    c0: float = Field(
        default=5.0,
        description="""Weighting parameter for m00 for ellipticity definition.
        """,
    )
    c2: float = Field(
        default=22.0,
        description="""Weighting parameter for m20 for ellipticity definition.
        """,
    )
    alpha: float = Field(
        default=1.0,
        description="""Power parameter for m00 for ellipticity definition.
        """,
    )
    beta: float = Field(
        default=0.0,
        description="""Power parameter for m20 for ellipticity definition.
        """,
    )
