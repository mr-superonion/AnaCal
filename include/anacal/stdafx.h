#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include <complex>
#include <tuple>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <array>
#include <limits>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

namespace py = pybind11;

// Releases the GIL for the enclosing scope, but only when this thread
// actually holds it.  The measurement entry points (Task::process_image,
// detector::find_peaks, GaussFit::process_cell) all release the GIL
// around their compute, and they also call each other -- the conditional
// makes the nested release a no-op instead of a fatal double release.
// The compute they guard must not touch the Python C API beyond reading
// existing arrays through unchecked accessors (allocations, copies and
// destructions of Python objects all need the GIL).
class ScopedGilRelease {
public:
    ScopedGilRelease() {
        if (PyGILState_Check()) {
            release_.emplace();
        }
    }

private:
    std::optional<py::gil_scoped_release> release_;
};

// Pixel type of the science and noise planes as they enter AnaCal.
//
// Survey coadds are stored at single precision (LSST ``ExposureF``, Euclid
// VIS), so AnaCal reads those two planes in that type instead of asking the
// caller for a float64 copy of the whole image.  The widening to double
// happens once, in ``Image::set_r``, where the pixels are copied into the FFT
// buffer: every float32 value is exactly representable as a double, so nothing
// is lost, and the FFT, PSF deconvolution, Gaussian smoothing and the derived
// qnumber all run at double precision exactly as before.
//
// The PSF stays double: it is a small computed stamp, not measured pixels.
using pixel_t = float;

// Same pixel type, but for arguments that are written back in place.
//
// The default ``py::array_t<T>`` lets pybind convert whatever it is handed,
// and a converted array is a temporary copy -- so an in-place function given
// the wrong dtype would edit the copy and leave the caller's pixels untouched,
// silently doing nothing.  Leaving ``forcecast`` off makes a wrong dtype a
// loud TypeError instead.
using pixel_array_inplace = py::array_t<pixel_t, py::array::c_style>;

// Constant for sqrt(2.0)
inline constexpr double sqrt2 = 1.4142135623730951;
// Constatn for 1 / sqrt(2.0)
inline constexpr double one_over_sqrt2 = 0.7071067811865475;

// The detection image is smoothed wider than the measurement scale by exactly
// sqrt(2).  Detection (detector.h, task.h), the stand-alone band weights and
// the FPFS noise level (ngmix/fitting.h) must all use the SAME widened scale;
// computing it in one place makes that invariant structural.
inline constexpr double detection_sigma(double sigma_arcsec) {
    return sigma_arcsec * sqrt2;
}

struct BrightStar {
    float x;
    float y;
    float r;
};

struct Position {
    double y;
    double x;
};


