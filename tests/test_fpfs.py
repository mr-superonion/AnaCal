import gc
import time

import anacal
import fpfs
import galsim
import numpy as np

from . import mem_used, print_mem
from .fpfs import smooth

scale = 0.2
ngrid = 64
psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(e1=0.02, e2=-0.02)
psf_data = (
    psf_obj.shift(0.5 * scale, 0.5 * scale)
    .drawImage(nx=ngrid // 2, ny=ngrid // 2, scale=scale)
    .array
)

nord = 4
det_nrot = 4
sigma_as = 0.53

task = fpfs.image.measure_source(
    psf_data,
    pix_scale=scale,
    sigma_arcsec=sigma_as,
    nord=nord,
    det_nrot=det_nrot,
)


def test_convolve():
    gal_obj = (
        psf_obj.shift(-3.5, 2) * 2
        + psf_obj.shift(2, -1) * 4
        + psf_obj.shift(-2, -0.5) * 4
        + psf_obj.shift(-3.2, 0.5) * 6
    )
    gal_data = gal_obj.drawImage(nx=ngrid, ny=ngrid, scale=scale).array
    det_task = anacal.fpfs.FpfsDetect(
        scale=scale,
        sigma_arcsec=sigma_as,
        det_nrot=task.det_nrot,
        klim=task.klim / scale,
    )
    smooth_data = det_task.smooth_image(
        gal_array=gal_data, psf_array=psf_data, noise_array=np.zeros((1, 1))
    )
    smooth_data2 = smooth(task, gal_data, psf_data)
    np.testing.assert_almost_equal(smooth_data, smooth_data2)
    return


if __name__ == "__main__":
    test_convolve()
