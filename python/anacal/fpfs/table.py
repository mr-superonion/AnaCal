import fitsio
import numpy as np
from numpy.typing import NDArray

from .base import get_det_col_names, get_shapelets_col_names


class FpfsObject(object):
    def __init__(self, nord, det_nrot):
        self.nord = nord
        self.det_nrot = det_nrot

        self.colnames = []
        if self.nord >= 4:
            snames, _ = get_shapelets_col_names(self.nord)
            self.colnames = self.colnames + snames
        if self.det_nrot >= 4:
            dnames = get_det_col_names(self.det_nrot)
            self.colnames = self.colnames + dnames

        self.ncol = len(self.colnames)
        self.di = {
            element: index for index, element in enumerate(self.colnames)
        }
        return


class FpfsCatalog(FpfsObject):
    def __init__(
        self,
        array: NDArray[np.float64],
        nord: int = -1,
        det_nrot: int = -1,
        noise: None | NDArray[np.float64] = None,
    ):
        super().__init__(nord=nord, det_nrot=det_nrot)

        if not isinstance(array, np.ndarray):
            raise TypeError("Input array has a wrong type")
        self.array = np.atleast_2d(array)
        assert self.array.shape[1] == self.ncol, "colnames has different length"
        if noise is not None:
            if not isinstance(noise, np.ndarray):
                raise ValueError("Input noise has a wrong type")
            self.noise = np.atleast_2d(noise)
            if self.noise.shape != self.array.shape:
                raise ValueError("Input noise and array has different shape")
        else:
            self.noise = None
        return

    @classmethod
    def from_file(cls, filename):
        with fitsio.FITS(filename, "r") as fits:
            assert "source" in fits
            array = fits["source"].read()
            if "noise" in fits:
                noise = fits["noise"].read()
            else:
                noise = None
            header = fits[0].read_header()
            nord = header["nord"]
            det_nrot = header["det_nrot"]
        return cls(array, nord=nord, det_nrot=det_nrot, noise=noise)

    def write(self, filename):
        with fitsio.FITS(filename, "rw", clobber=True) as fits:
            # Write data to a new extension with a specific name
            fits.write(self.array, extname="source")
            if self.noise is not None:
                fits.write(self.noise, extname="noise")
            header = fits[0].read_header()
            header["nord"] = self.nord
            header["det_nrot"] = self.det_nrot
            header["colnames"] = self.colnames
            fits[0].write_keys(header)
        return


class FpfsCovariance(FpfsObject):
    def __init__(
        self,
        array: NDArray,
        nord: int = -1,
        det_nrot: int = -1,
    ):
        super().__init__(nord=nord, det_nrot=det_nrot)

        if not isinstance(array, np.ndarray):
            raise TypeError("Input array has a wrong type")
        self.array = np.atleast_2d(array)
        assert self.array.shape[1] == self.ncol, "array has wrong shape"

        self.std_modes = np.sqrt(np.diagonal(self.array))
        if nord >= 4 and "m00" in self.colnames:
            self.std_m00 = self.std_modes[self.di["m00"]]
        if nord >= 4 and "m20" in self.colnames:
            self.std_r2 = np.sqrt(
                self.array[self.di["m00"], self.di["m00"]]
                + self.array[self.di["m20"], self.di["m20"]]
                + self.array[self.di["m00"], self.di["m20"]]
                + self.array[self.di["m20"], self.di["m00"]]
            )
        det_name_list = ["v%d" % _ for _ in range(self.det_nrot)]
        if det_nrot >= 4 and set(det_name_list).issubset(self.colnames):
            self.std_v = np.average(
                np.array(
                    [
                        self.std_modes[self.di["v%d" % _]]
                        for _ in range(self.det_nrot)
                    ]
                )
            )
        return

    @classmethod
    def from_file(cls, filename):
        with fitsio.FITS(filename, "r") as fits:
            assert "covariance" in fits
            array = fits["covariance"].read()
            header = fits[0].read_header()
            nord = header["nord"]
            det_nrot = header["det_nrot"]
        return cls(array, nord=nord, det_nrot=det_nrot)

    def write(self, filename):
        with fitsio.FITS(filename, "rw", clobber=True) as fits:
            # Write data to a new extension with a specific name
            fits.write(self.array, extname="covariance")
            header = fits[0].read_header()
            header["nord"] = self.nord
            header["det_nrot"] = self.det_nrot
            header["colnames"] = self.colnames
            fits[0].write_keys(header)
        return
