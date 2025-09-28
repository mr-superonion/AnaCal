# AnaCal
[![docs](https://readthedocs.org/projects/anacal/badge/?version=latest)](https://anacal.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/mr-superonion/AnaCal/actions/workflows/tests.yml/badge.svg)](https://github.com/mr-superonion/AnaCal/actions/workflows/tests.yml)
[![pypi](https://img.shields.io/pypi/v/anacal)](https://pypi.org/project/anacal/)
[![conda-forge](https://anaconda.org/conda-forge/anacal/badges/version.svg)](https://anaconda.org/conda-forge/anacal)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Analytic Calibration for Perturbation Estimation from Galaxy Images.

This framework is designed to measure the shear responses of both existing and future shape estimators. 
Our goal is to develop a suite of analytical shear estimators that can infer shear with sub-percent accuracy 
while remaining computationally efficient.

To compute shear response, we introduce the concept of 
[pixel shear response](https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.4904L/abstract)---the derivatives 
of pixel values with respect to applied shear distortions. We then propagate these responses using 
[quintuple numbers](https://ui.adsabs.harvard.edu/abs/2025arXiv250616607L/abstract), a technique for efficient
shear response tracking. For accurate noise bias correction, we adopt a [renoising
approach](https://ui.adsabs.harvard.edu/abs/2025MNRAS.536.3663L/abstract) that enables analytical treatment
of noise effects.

Currently, the framework supports the following analytical shear estimators:
+ [FPFS](https://github.com/mr-superonion/FPFS): A fixed moments method based
  on shapelets including analytic correction for selection, detection and noise
  bias. (see [ref1](https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.4445L/abstract),
  [ref2](https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.4850L/abstract),
  [ref3](https://ui.adsabs.harvard.edu/abs/2024MNRAS.52710388L/abstract),
  and [ref4](https://ui.adsabs.harvard.edu/abs/2025MNRAS.537..507P/abstract).)
+ [NGMIX](https://github.com/esheldon/ngmix): Gaussian model fitting.
  (see [ref5](https://ui.adsabs.harvard.edu/abs/2025arXiv250616607L/abstract))

## Installation
Users can clone this repository and install the latest package by
```shell
git clone https://github.com/mr-superonion/AnaCal.git
cd AnaCal
# install required softwares
conda install -c conda-forge --file requirements.txt
# install required softwares for unit tests (if necessary)
conda install -c conda-forge --file requirements_test.txt
pip install . --user
```
or install stable version
```
pip install anacal
```
or
```
conda install -c conda-forge anacal
```

## Examples
Examples can be found [here](https://github.com/mr-superonion/AnaCal/blob/main/examples/).

## Development

Before sending pull request, please make sure that the modified code passed the
pytest and flake8 tests. Run the following commands under the root directory
for the tests:

```shell
flake8
pytest -vv
```
----
