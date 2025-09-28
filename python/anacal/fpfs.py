import numpy as np
import numpy.lib.recfunctions as rfn
from ._anacal.fpfs import (
    FpfsImage,
)
from ._anacal.fpfs import detlets2d as _detlets2d
from ._anacal.fpfs import (
    fpfs_cut_sigma_ratio,
    fpfs_det_sigma2,
)
from ._anacal.fpfs import gauss_kernel_rfft as _gauss_kernel_rfft
from ._anacal.fpfs import get_kmax as _get_kmax
from ._anacal.fpfs import (
    measure_fpfs,
    measure_fpfs_shape,
    measure_fpfs_wdet,
    measure_fpfs_wdet0,
    measure_fpfs_wsel,
    measure_shapelets_dg,
)
from ._anacal.fpfs import shapelets2d as _shapelets2d
from ._anacal.fpfs import shapelets2d_func as _shapelets2d_func
from ._anacal.image import Image
from numpy.typing import NDArray
from .psf import BasePsf
from pydantic import BaseModel, Field

npix_patch = 256
npix_overlap = 64


# M_{nm}
# nm = n*(norder+1)+m
# This setup is able to derive kappa response and shear response
# Only uses M00, M20, M22 (real and img), M40, M42(real and img), M60
norder_shapelets = 6
name_s = [
    "m00",
    "m20",
    "m22c",
    "m22s",
    "m40",
    "m42c",
    "m42s",
    "m44c",
    "m44s",
    "m60",
    "m64c",
    "m64s",
]
name_d = [
    "v0",
    "v1",
    "v2",
    "v3",
    "v0r1",
    "v1r1",
    "v2r1",
    "v3r1",
    "v0r2",
    "v1r2",
    "v2r2",
    "v3r2",
]


class FpfsKernel:
    """Fpfs measurement kernel object

    Args:
    npix (int): number of pixels in a postage stamp
    pixel_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Shapelet kernel size
    kmax (float | None): maximum k
    psf_array (ndarray): an average PSF image [default: None]
    kmax_thres (float): the tuncation threshold on Gaussian [default: 1e-20]
    do_detection (bool): whether compute detection kernel
    """

    def __init__(
        self,
        *,
        npix: int,
        pixel_scale: float,
        sigma_arcsec: float,
        kmax: float | None = None,
        psf_array: NDArray | None = None,
        kmax_thres: float = 1e-20,
        do_detection: bool = True,
        verbose=False,
    ) -> None:
        super().__init__(verbose=verbose)
        self.npix = npix
        self.do_detection = do_detection

        self.sigma_arcsec = sigma_arcsec
        if self.sigma_arcsec > 3.0:
            raise ValueError("sigma_arcsec should be < 3 arcsec")

        # A few import scales
        self.pixel_scale = pixel_scale
        self._dk = 2.0 * np.pi / self.npix  # assuming pixel scale is 1

        # the following two assumes pixel_scale = 1
        self.sigmaf = float(self.pixel_scale / self.sigma_arcsec)
        if psf_array is None:
            # make a delta psf
            psf_array = np.zeros((npix, npix))
            psf_array[npix // 2, npix // 2] = 1
        else:
            if not psf_array.shape == (npix, npix):
                raise ValueError("psf arry has a wrong shape")

        psf_f = np.fft.rfft2(psf_array)
        self.psf_array = psf_array
        self.psf_pow = (np.abs(psf_f) ** 2.0).astype(np.float64)
        if kmax is None:
            assert psf_array is not None
            # truncation raduis for PSF in Fourier space
            self.kmax = (
                get_kmax(
                    psf_pow=self.psf_pow,
                    sigma=self.sigmaf / np.sqrt(2.0),
                    kmax_thres=kmax_thres,
                )
                * self._dk
            )
        else:
            self.kmax = kmax

        self.prepare_fpfs_bases()
        return

    def prepare_fpfs_bases(self):
        """This fucntion prepare the FPFS bases (shapelets and detectlets)"""
        bfunc = []
        self.colnames = []
        sfunc, snames = shapelets2d(
            norder=norder_shapelets,
            npix=self.npix,
            sigma=self.sigmaf,
            kmax=self.kmax,
        )
        bfunc.append(sfunc)
        self.colnames = self.colnames + snames
        if self.do_detection:
            dfunc, dnames = detlets2d(
                npix=self.npix,
                sigma=self.sigmaf,
                kmax=self.kmax,
            )
            bfunc.append(dfunc)
            self.colnames = self.colnames + dnames
        self.bfunc = np.vstack(bfunc)
        self.ncol = len(self.colnames)
        self.dtype = [(name, "f8") for name in self.colnames]
        self.bfunc_use = np.transpose(self.bfunc, (1, 2, 0))
        self.di = {
            element: index for index, element in enumerate(self.colnames)
        }
        return

    def prepare_covariance(
        self, variance: float, noise_pf: NDArray | None = None
    ):
        """Estimate covariance of measurement error

        Args:
        variance (float): Noise variance
        noise_pf (NDArray | None): Power spectrum (assuming homogeneous) of
        noise
        """
        variance = variance * 2.0
        if noise_pf is not None:
            if noise_pf.shape == (self.npix, self.npix // 2 + 1):
                # rfft
                noise_pf = np.array(noise_pf, dtype=np.float64)
            elif noise_pf.shape == (self.npix, self.npix):
                # fft
                noise_pf = np.fft.ifftshift(noise_pf)
                noise_pf = np.array(
                    noise_pf[:, : self.npix // 2 + 1], dtype=np.float64
                )
            else:
                raise ValueError("noise power not in correct shape")
        else:
            ss = (self.npix, self.npix // 2 + 1)
            noise_pf = np.ones(ss)
        norm_factor = variance * self.npix**2.0 / noise_pf[0, 0]
        noise_pf = noise_pf * norm_factor

        img_obj = Image(nx=self.npix, ny=self.npix, scale=self.pixel_scale)
        img_obj.set_f(noise_pf)
        img_obj.deconvolve(
            psf_image=self.psf_pow,
            klim=self.kmax / self.pixel_scale,
        )
        noise_pf_deconv = img_obj.draw_f().real
        del img_obj

        _w = np.ones(self.psf_pow.shape) * 2.0
        _w[:, 0] = 1.0
        _w[:, -1] = 1.0
        cov_elems = (
            np.tensordot(
                self.bfunc * (_w * noise_pf_deconv)[np.newaxis, :, :],
                np.conjugate(self.bfunc),
                axes=((1, 2), (1, 2)),
            ).real
            / self.pixel_scale**4.0
        )
        self.std_modes = np.sqrt(np.diagonal(cov_elems))
        self.std_m00 = self.std_modes[self.di["m00"]]
        return cov_elems


def gauss_kernel_rfft(
    ny: int, nx: int, sigma: float, kmax: float, return_grid: bool = False
):
    """Generate a Gaussian kernel on grids for :func:`numpy.fft.rfft`.

    This function is provided for backwards compatibility with the original
    NumPy implementation but now delegates to the high-performance C++
    extension.
    """

    return _gauss_kernel_rfft(ny, nx, sigma, kmax, return_grid)


def shapelets2d_func(npix: int, norder: int, sigma: float, kmax: float):
    """Generate complex shapelet functions in Fourier space."""

    return _shapelets2d_func(npix, norder, sigma, kmax)


def shapelets2d(norder: int, npix: int, sigma: float, kmax: float):
    """Generate real-valued shapelet functions in Fourier space."""

    chi = _shapelets2d(norder, npix, sigma, kmax)
    return np.array(chi), name_s


def detlets2d(
    npix: int,
    sigma: float,
    kmax: float,
):
    """Generate complex detection basis functions in Fourier space."""

    psi = _detlets2d(npix, sigma, kmax)
    return psi, name_d


def get_kmax(
    psf_pow: NDArray,
    sigma: float,
    kmax_thres: float = 1e-20,
) -> float:
    """Estimate the truncation radius ``kmax`` for the Gaussian kernel."""

    return _get_kmax(psf_pow, sigma, kmax_thres)


class FpfsTask(FpfsKernel):
    """A base class for measurement

    Args:
    npix (int): number of pixels in a postage stamp
    pixel_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Shapelet kernel size
    noise_variance (float): variance of image noise
    kmax (float | None): maximum k
    psf_array (ndarray): an average PSF image [default: None]
    kmax_thres (float): the tuncation threshold on Gaussian [default: 1e-20]
    do_detection (bool): whether compute detection kernels
    bound (int): minimum distance to boundary [default: 0]
    verbose (bool): whether display INFO
    """

    def __init__(
        self,
        *,
        npix: int,
        pixel_scale: float,
        sigma_arcsec: float,
        noise_variance: float = -1,
        kmax: float | None = None,
        psf_array: NDArray | None = None,
        kmax_thres: float = 1e-20,
        do_detection: bool = True,
        bound: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            npix=npix,
            pixel_scale=pixel_scale,
            sigma_arcsec=sigma_arcsec,
            kmax=kmax,
            psf_array=psf_array,
            kmax_thres=kmax_thres,
            do_detection=do_detection,
            verbose=verbose,
        )
        klim = 1e10 if self.kmax is None else (self.kmax / self.pixel_scale)
        self.prepare_fpfs_bases()
        self.mtask = FpfsImage(
            nx=self.npix,
            ny=self.npix,
            scale=self.pixel_scale,
            sigma_arcsec=self.sigma_arcsec,
            klim=klim,
            psf_array=self.psf_array,
            use_estimate=True,
        )

        if self.do_detection:
            self.dtask = FpfsImage(
                nx=npix_patch,
                ny=npix_patch,
                scale=self.pixel_scale,
                sigma_arcsec=self.sigma_arcsec,
                klim=klim,
                psf_array=self.psf_array,
                use_estimate=True,
                npix_overlap=npix_overlap,
                bound=bound,
            )
            if not noise_variance > 0:
                raise ValueError("Noise variance should be positive")
            self.prepare_covariance(variance=noise_variance)
        else:
            self.dtask = None

        return

    def detect(
        self,
        *,
        gal_array: NDArray,
        fthres: float,
        pthres: float,
        omega_v: float,
        v_min: float,
        noise_array: NDArray | None = None,
        mask_array: NDArray | None = None,
        star_cat: NDArray | None = None,
    ) -> NDArray:
        """This function detects galaxy from image

        Args:
        gal_array (NDArray): galaxy image data
        fthres (float): flux threshold
        pthres (float): peak threshold
        omega_v (float): smoothness parameter for pixel difference
        noise_array (NDArray|None): pure noise image
        mask_array (NDArray|None): mask image
        star_cat (NDArray|None): bright star catalog

        Returns:
        (NDArray): galaxy detection catalog
        """
        assert self.dtask is not None
        return self.dtask.detect_source(
            gal_array=gal_array,
            fthres=fthres,
            pthres=pthres,
            std_m00=self.std_m00 * self.pixel_scale**2.0,
            omega_v=omega_v * self.pixel_scale**2.0,
            v_min=v_min * self.pixel_scale**2.0,
            noise_array=noise_array,
            mask_array=mask_array,
        )

    def run_psf_array(
        self,
        *,
        gal_array: NDArray,
        psf_array: NDArray,
        det: NDArray | None = None,
        noise_array: NDArray | None = None,
    ) -> tuple[NDArray, NDArray | None]:
        """This function measure galaxy shapes at the position of the detection
        using PSF image data

        Args:
        gal_array (NDArray): galaxy image data
        psf_array (NDArray): psf image data
        det (list|None): detection catalog
        noise_array (NDArray | None): noise image data [default: None]

        Returns:
        src_g (NDArray): source measurement catalog
        src_n (NDArray): noise measurement catalog
        """
        # self.logger.warning("Input PSF is array")
        src_g = self.mtask.measure_source(
            gal_array=gal_array,
            filter_image=self.bfunc_use,
            psf_array=psf_array,
            det=det,
            do_rotate=False,
        )
        if noise_array is not None:
            src_n = self.mtask.measure_source(
                gal_array=noise_array,
                filter_image=self.bfunc_use,
                psf_array=psf_array,
                det=det,
                do_rotate=True,
            )
            src_g = src_g + src_n
        else:
            src_n = None
        return src_g, src_n

    def run_psf_python(
        self,
        gal_array: NDArray,
        psf_obj: BasePsf,
        det: NDArray,
        noise_array: NDArray | None = None,
    ) -> tuple[NDArray, NDArray | None]:
        """This function measure galaxy shapes at the position of the detection
        using PSF image data

        Args:
        gal_array (NDArray): galaxy image data
        psf_obj (BasePsf): PSF object in python
        noise_array (NDArray | None): noise image data [default: None]
        det (list|None): detection catalog

        Returns:
        src_g (NDArray): source measurement catalog
        src_n (NDArray): noise measurement catalog
        """
        # self.logger.warning("Input PSF is python object")
        src_g = []
        src_n = []
        for _d in det:
            this_psf_array = psf_obj.draw(x=_d["x"], y=_d["y"])
            srow = self.mtask.measure_source_at(
                gal_array=gal_array,
                filter_image=self.bfunc_use,
                psf_array=this_psf_array,
                y=_d["y"],
                x=_d["x"],
                do_rotate=False,
            )
            if noise_array is not None:
                nrow = self.mtask.measure_source_at(
                    gal_array=noise_array,
                    filter_image=self.bfunc_use,
                    psf_array=this_psf_array,
                    y=_d["y"],
                    x=_d["x"],
                    do_rotate=True,
                )
                srow = srow + nrow
                src_n.append(nrow)
            src_g.append(srow)
        if len(src_n) == 0:
            src_n = None
        else:
            assert len(src_n) == len(src_g)
            src_n = np.array(src_n)
        src_g = np.array(src_g)
        return src_g, src_n


    def run(
        self,
        gal_array: NDArray,
        psf: BasePsf | NDArray,
        det: NDArray | None = None,
        noise_array: NDArray | None = None,
    ):
        """This function measure galaxy shapes at the position of the detection

        Args:
        gal_array (NDArray): galaxy image data
        det (NDArray): detection catalog
        psf (BasePsf | NDArray): psf image data or psf model
        noise_array (NDArray | None): noise image data [default: None]

        Returns:
        (NDArray): galaxy measurement catalog
        """
        if isinstance(psf, np.ndarray):
            src_g, src_n = self.run_psf_array(
                gal_array=gal_array,
                psf_array=psf,
                noise_array=noise_array,
                det=det,
            )
        elif isinstance(psf, BasePsf):
            assert det is not None
            # For the case PSF is a Python object
            src_g, src_n = self.run_psf_python(
                gal_array=gal_array,
                psf_obj=psf,
                noise_array=noise_array,
                det=det,
            )
        else:
            raise RuntimeError("psf does not have a correct type")
        src_g = rfn.unstructured_to_structured(
            arr=src_g,
            dtype=self.dtype,
        )
        if src_n is not None:
            src_n = rfn.unstructured_to_structured(arr=src_n, dtype=self.dtype)

        return {
            "data": src_g,
            "noise": src_n,
        }


class FpfsConfig(BaseModel):
    npix: int = Field(
        default=64,
        description="""size of the stamp before Fourier Transform
        """,
    )
    kmax_thres: float = Field(
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
    sigma_arcsec1: float = Field(
        default=-1,
        description="""Smoothing scale of the second shapelet kernel.
        """,
    )
    sigma_arcsec2: float = Field(
        default=-1,
        description="""Smoothing scale of the third shapelet kernel.
        """,
    )
    pthres: float = Field(
        default=0.12,
        description="""Detection threshold (peak identification) for the
        pooling.
        """,
    )
    fthres: float = Field(
        default=8.0,
        description="""Detection threshold (minimum signal-to-noise ratio) for
        the first pooling.
        """,
    )
    omega_r2: float = Field(
        default=4.8,
        description="""
        smoothness parameter for r2 cut
        """,
    )
    r2_min: float = Field(
        default=0.1,
        description="""Minimum trace moment matrix
        """,
    )
    omega_v: float = Field(
        default=0.9,
        description="""
        smoothness parameter for v cut
        """,
    )
    v_min: float = Field(
        default=0.45,
        description="""Minimum of v
        """,
    )
    snr_min: float = Field(
        default=12,
        description="""Minimum Signal-to-Noise Ratio for detection.
        """,
    )
    c0: float = Field(
        default=8.4,
        description="""Weighting parameter for m00 for ellipticity definition.
        """,
    )


def process_image(
    *,
    fpfs_config: FpfsConfig,
    pixel_scale: float,
    noise_variance: float,
    mag_zero: float,
    gal_array: NDArray,
    psf_array: NDArray,
    noise_array: NDArray | None = None,
    mask_array: NDArray | None = None,
    star_catalog: NDArray | None = None,
    detection: NDArray | None = None,
    psf_object: BasePsf | None = None,
    do_compute_detect_weight: bool = True,
    only_return_detection_modes: bool = False,
    base_column_name: str | None = None,
):
    """Run measurement algorithms on the input exposure, and optionally
    populate the resulting catalog with extra information.

    Args:
    fpfs_config (FpfsConfig):  configuration object
    pixel_scale (float): pixel scale in arcsec
    noise_variance (float): variance of image noise
    mag_zero (float): magnitude zero point
    do_detection (bool): whether compute detection kernel
    gal_array (NDArray[float64]): galaxy exposure array
    psf_array (ndarray): an average PSF image
    noise_array (NDArray | None): pure noise array
    mask_array (NDArray | None): mask array (1 for masked)
    star_catalog (NDArray | None): bright star catalog
    detection (NDArray | None): detection catalog
    psf_object (BasePsf | None): PSF object
    do_compute_detect_weight (bool): whether to compute detection weight
    only_return_detection_modes (bool): only return linear modes for detection
    base_column_name (str | None): base column name

    Returns:
    (NDArray) FPFS catalog
    """
    if only_return_detection_modes:
        assert do_compute_detect_weight

    ratio = 1.0 / (10 ** ((30 - mag_zero) / 2.5))
    r2_min = fpfs_config.r2_min * ratio
    omega_r2 = fpfs_config.omega_r2 * ratio
    v_min = fpfs_config.v_min * ratio
    omega_v = fpfs_config.omega_v * ratio
    fpfs_c0 = fpfs_config.c0 * ratio

    if psf_object is None:
        psf_object = psf_array

    out_list = []

    if do_compute_detect_weight or (detection is None):
        ftask = FpfsTask(
            npix=fpfs_config.npix,
            pixel_scale=pixel_scale,
            sigma_arcsec=fpfs_config.sigma_arcsec,
            noise_variance=noise_variance,
            psf_array=psf_array,
            kmax_thres=fpfs_config.kmax_thres,
            do_detection=True,
            bound=fpfs_config.bound,
        )
        std_m00 = ftask.std_m00
        m00_min = fpfs_config.snr_min * std_m00
        if detection is None:
            detection = ftask.detect(
                gal_array=gal_array,
                fthres=fpfs_config.fthres,
                pthres=fpfs_config.pthres,
                noise_array=noise_array,
                v_min=v_min,
                omega_v=omega_v,
                mask_array=mask_array,
                star_cat=star_catalog,
            )
        else:
            colnames = ("y", "x")
            if detection.dtype.names != colnames:
                raise ValueError("detection has wrong cloumn names")
        out_list.append(detection)

        if do_compute_detect_weight:
            src = ftask.run(
                gal_array=gal_array,
                psf=psf_object,
                det=detection,
                noise_array=noise_array,
            )
            if only_return_detection_modes:
                return src
            meas = measure_fpfs(
                C0=fpfs_c0,
                v_min=v_min,
                omega_v=omega_v,
                pthres=fpfs_config.pthres,
                m00_min=m00_min,
                std_m00=std_m00,
                r2_min=r2_min,
                omega_r2=omega_r2,
                x_array=src["data"],
                y_array=src["noise"],
            )
            del src
            map_dict = {name: "fpfs_" + name for name in meas.dtype.names}
            out_list.append(rfn.rename_fields(meas, map_dict))

        del ftask

    if fpfs_config.sigma_arcsec1 > 0:
        ftask = FpfsTask(
            npix=fpfs_config.npix,
            pixel_scale=pixel_scale,
            sigma_arcsec=fpfs_config.sigma_arcsec1,
            psf_array=psf_array,
            kmax_thres=fpfs_config.kmax_thres,
            do_detection=False,
        )
        src = ftask.run(
            gal_array=gal_array,
            psf=psf_object,
            det=detection,
            noise_array=noise_array,
        )
        meas1 = measure_fpfs(
            C0=fpfs_c0,
            x_array=src["data"],
            y_array=src["noise"],
        )
        del src, ftask
        map_dict = {name: "fpfs1_" + name for name in meas1.dtype.names}
        out_list.append(rfn.rename_fields(meas1, map_dict))
        del meas1

    if fpfs_config.sigma_arcsec2 > 0:
        ftask = FpfsTask(
            npix=fpfs_config.npix,
            pixel_scale=pixel_scale,
            sigma_arcsec=fpfs_config.sigma_arcsec2,
            psf_array=psf_array,
            kmax_thres=fpfs_config.kmax_thres,
            do_detection=False,
        )
        src = ftask.run(
            gal_array=gal_array,
            psf=psf_object,
            det=detection,
            noise_array=noise_array,
        )
        meas2 = measure_fpfs(
            C0=fpfs_c0,
            x_array=src["data"],
            y_array=src["noise"],
        )
        del src, ftask
        map_dict = {name: "fpfs2_" + name for name in meas2.dtype.names}
        out_list.append(rfn.rename_fields(meas2, map_dict))
        del meas2

    result = rfn.merge_arrays(
        out_list,
        flatten=True,
        usemask=False,
    )
    if base_column_name is not None:
        assert result.dtype.names is not None
        map_dict = {
            name: base_column_name + name for name in result.dtype.names
        }
        result = rfn.rename_fields(result, map_dict)

    return result
