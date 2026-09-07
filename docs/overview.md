# Overview

AnaCal (**Ana**lytic **Cal**ibration) is a Python/C++ library for accurate
measurement of perturbations on galaxy images.  It provides analytic shear
estimators capable of inferring shear with sub-percent accuracy while
maintaining minimal computational cost.

## Features

- **FPFS shear estimator** — measures shapelet and detection modes in Fourier
  space, with analytic noise-bias correction.
- **Flexible PSF handling** — supports both array-based and spatially varying
  PSF models via the `BasePsf` interface.
- **High-performance C++ core** — critical inner loops (shapelet convolution,
  source detection, deconvolution) are implemented in C++ and exposed through
  pybind11.

## Dependencies

- Python ≥ 3.10
- [NumPy](https://numpy.org/)
- [fitsio](https://github.com/esheldon/fitsio)

## Installation

### Stable release

```shell
pip install anacal
```

### From source

```shell
git clone https://github.com/mr-superonion/AnaCal.git
cd AnaCal
pip install .
```

### Development install

```shell
git clone https://github.com/mr-superonion/AnaCal.git
cd AnaCal
pip install -e ".[dev]"
```

## Development

Before sending a pull request, please make sure that the modified code passes
the pytest and flake8 tests:

```shell
flake8
pytest -vv
```
