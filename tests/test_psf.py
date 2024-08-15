import anacal
import numpy as np


class my_psf(anacal.psf.PyPsf):

    def __init__(self, a):
        super().__init__()
        self.a = a

    def draw(self, x, y):
        return np.ones((10, 10))


def test_pypsf():
    psf_obj = my_psf(a=1)
    assert psf_obj.crun is False
    return


if __name__ == "__main__":
    test_pypsf()
