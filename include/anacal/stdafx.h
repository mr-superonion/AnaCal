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

namespace py = pybind11;

// Constatn for 1 / sqrt(2.0)
inline constexpr double one_over_sqrt2 = 0.7071067811865475;

struct BrightStar {
    float x;
    float y;
    float r;
};

struct FpfsPeaks {
    int y;
    int x;
    int is_peak;
    int mask_value;
};


struct FpfsShapelets {
    double m00;
    double m20;
    double m22c;
    double m22s;
    double m40;
    double m42c;
    double m42s;
    double m44c;
    double m44s;
    double m60;
    double m64c;
    double m64s;
};

struct FpfsDetect {
    // Fields from FpfsShapelets
    double m00;
    double m20;
    double m22c;
    double m22s;
    double m40;
    double m42c;
    double m42s;
    double m44c;
    double m44s;
    double m60;
    double m64c;
    double m64s;

    // Additional fields specific to FpfsDetect
    double v0;
    double v1;
    double v2;
    double v3;
    double v0_g1;
    double v1_g1;
    double v2_g1;
    double v3_g1;
    double v0_g2;
    double v1_g2;
    double v2_g2;
    double v3_g2;
};

struct FpfsShapeletsResponse {
    double m00_g1;
    double m00_g2;
    double m20_g1;
    double m20_g2;
    double m22c_g1;
    double m22s_g2;
    double m42c_g1;
    double m42s_g2;
};


struct FpfsShapeCatalog {
    double e1;
    double e1_g1;
    double e2_g2;
    double q1;
    double q1_g1;
    double q2_g2;
};


struct FpfsWeight {
    double w;
    double w_g1;
    double w_g2;
};
