# AnaCal
[![Python application](https://github.com/mr-superonion/AnaCal/actions/workflows/python-app.yml/badge.svg)](https://github.com/mr-superonion/AnaCal/actions/workflows/python-app.yml)
[![pypi](https://github.com/mr-superonion/AnaCal/actions/workflows/pypi.yml/badge.svg)](https://github.com/mr-superonion/AnaCal/actions/workflows/pypi.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Analytic Calibration for Perturbation Estimation from Galaxy Images.

This framework is devised to bridge various analytic shear estimators that have
been developed or are anticipated to be created in the future. We intend to
develop a suite of analytical shear estimators capable of inferring shear with
subpercent accuracy, all while maintaining minimal computational time. The
currently supported analytic shear estimators are:
+ [FPFS](https://github.com/mr-superonion/FPFS)

## Installation
Users can clone this repository and install the latest package by
```
pip install .
```
or install stable verion from pypi
```
pip install anacal
```

## Examples
Examples can be found [here](./docs/examples).

## Development

Before sending pull request, please make sure that the modified code passed the
pytest and flake8 tests. Run the following commands under the root directory
for the tests:

```shell
flake8
pytest -vv
```

----
