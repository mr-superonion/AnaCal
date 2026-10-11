#include "anacal.h"

namespace anacal {
namespace renoise {

namespace {

inline Mode
mode_from_string(const std::string& mode) {
    if (mode == "rotate") return Mode::rotate;
    if (mode == "spectral") return Mode::spectral;
    throw std::invalid_argument(
        "renoise mode must be 'rotate' or 'spectral', got '" + mode + "'"
    );
}


py::array_t<double>
half_plane_array(const std::vector<double>& v, int n) {
    const int kxl = half_width(n);
    py::array_t<double> out({n, kxl});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

std::vector<double>
half_plane_vector(const py::array_t<double>& arr, int& n) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument("expected a (n, n/2 + 1) array");
    }
    n = static_cast<int>(arr.shape(0));
    if (static_cast<int>(arr.shape(1)) != half_width(n)) {
        throw std::invalid_argument("expected a (n, n/2 + 1) array");
    }
    auto r = arr.unchecked<2>();
    std::vector<double> v(static_cast<std::size_t>(n) * half_width(n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < half_width(n); ++i) {
            v[static_cast<std::size_t>(j) * half_width(n) + i] = r(j, i);
        }
    }
    return v;
}

} // namespace

void
pyExportRenoise(py::module_& m) {
    py::module_ rn = m.def_submodule(
        "renoise",
        "minimal-variance renoising: noise spectrum, spin-2 symmetrization "
        "and the added-noise / assigned-response kernels (rfft half-plane "
        "layout (n, n/2 + 1), as numpy.fft.rfft2)"
    );

    py::class_<NoiseSpectrum>(rn, "NoiseSpectrum")
        .def_readonly("n", &NoiseSpectrum::n)
        .def_readonly("scale", &NoiseSpectrum::scale)
        .def_readonly("white", &NoiseSpectrum::white)
        .def_readonly("variance", &NoiseSpectrum::variance)
        .def_readonly("symmetrized", &NoiseSpectrum::symmetrized)
        .def_property_readonly("S_raw", [](const NoiseSpectrum& s) {
            return half_plane_array(s.S_raw, s.n);
        })
        .def_property_readonly("g", [](const NoiseSpectrum& s) {
            if (!s.symmetrized) {
                throw std::runtime_error(
                    "g is only defined for a symmetrized spectrum"
                );
            }
            return half_plane_array(s.g, s.n);
        })
        .def_property_readonly("S", [](const NoiseSpectrum& s) {
            return half_plane_array(s.S, s.n);
        })
        .def_property_readonly("S_R", [](const NoiseSpectrum& s) {
            return half_plane_array(s.S_R, s.n);
        })
        .def_property_readonly("dS1", [](const NoiseSpectrum& s) {
            return half_plane_array(s.dS1, s.n);
        })
        .def_property_readonly("dS2", [](const NoiseSpectrum& s) {
            return half_plane_array(s.dS2, s.n);
        });

    rn.def("make_noise_spectrum", &make_noise_spectrum,
        "Image-noise power spectrum of one patch and band on an n x n grid: "
        "S(k), S(R k) and the shear derivatives (k1 d1 - k2 d2) S and "
        "(k1 d2 + k2 d1) S, exact from transforms of x_j C(x).  corr is the "
        "noise correlation function with lag 0 at pixel (ny // 2, nx // 2); "
        "it sets the shape of the spectrum and its amplitude is rescaled so "
        "that C(0) = variance.  None means white noise of the given "
        "variance.",
        py::arg("n"), py::arg("scale"), py::arg("variance"),
        py::arg("corr") = py::none()
    );

    py::class_<RenoiseKernels>(rn, "RenoiseKernels")
        .def_property_readonly("mode", [](const RenoiseKernels& k) {
            return std::string(k.mode == Mode::rotate ? "rotate" : "spectral");
        })
        .def_property_readonly("Sd_rot", [](const RenoiseKernels& k) {
            return half_plane_array(k.Sd_rot, k.n);
        })
        .def_readonly("n", &RenoiseKernels::n)
        .def_readonly("scale", &RenoiseKernels::scale)
        .def_readonly("klim", &RenoiseKernels::klim)
        .def_readonly("c", &RenoiseKernels::c)
        .def_readonly("delta", &RenoiseKernels::delta)
        .def_property_readonly("Sd", [](const RenoiseKernels& k) {
            return half_plane_array(k.Sd, k.n);
        })
        .def_property_readonly("Sd_sym", [](const RenoiseKernels& k) {
            return half_plane_array(k.Sd_sym, k.n);
        })
        .def_property_readonly("a_sym1", [](const RenoiseKernels& k) {
            return half_plane_array(k.a_sym1, k.n);
        })
        .def_property_readonly("a_sym2", [](const RenoiseKernels& k) {
            return half_plane_array(k.a_sym2, k.n);
        })
        .def_property_readonly("g", [](const RenoiseKernels& k) {
            return half_plane_array(k.g, k.n);
        })
        .def_property_readonly("w", [](const RenoiseKernels& k) {
            return half_plane_array(k.w, k.n);
        })
        .def_property_readonly("kappa1", [](const RenoiseKernels& k) {
            return half_plane_array(k.kappa1, k.n);
        })
        .def_property_readonly("kappa2", [](const RenoiseKernels& k) {
            return half_plane_array(k.kappa2, k.n);
        })
        .def_property_readonly("ap1", [](const RenoiseKernels& k) {
            return half_plane_array(k.ap1, k.n);
        })
        .def_property_readonly("ap2", [](const RenoiseKernels& k) {
            return half_plane_array(k.ap2, k.n);
        })
        .def_property_readonly("Sd0", [](const RenoiseKernels& k) {
            return half_plane_array(k.Sd0, k.n);
        })
        .def_readonly("noise_rotated", &RenoiseKernels::noise_rotated);

    rn.def("symmetrize_spectrum", &symmetrize_spectrum,
        "The spectrum of the image noise after the spin-2 filler of the "
        "patch PSF: S_s = S S^d_sym / S^d, with its shear derivatives; "
        "S_raw keeps the pure-noise spectrum and g is the filler filter "
        "on this grid.",
        py::arg("noise_spectrum"), py::arg("psf"), py::arg("klim"),
        py::arg("delta") = 0.05
    );

    rn.def("make_renoise_kernels",
        [](const NoiseSpectrum& ns, const py::array_t<double>& psf,
           double klim, double c, double delta, bool symmetrize,
           bool noise_rotated, const std::string& mode) {
            return make_renoise_kernels(
                ns, psf, klim, c, delta, symmetrize, noise_rotated,
                mode_from_string(mode)
            );
        },
        "Per-PSF kernels of the minimal-variance renoising: the "
        "deconvolved spectrum S^d, its 90-degree-symmetrized smooth "
        "maximum S^d_sym, the shear derivatives a_sym1 / a_sym2 of "
        "ln S^d_sym, the filler filter g = sqrt(S^d_sym / S^d - 1), the "
        "added-noise amplitude w = sqrt(c S^d_sym / S^d) and the "
        "assigned-response filters kappa_j = -sqrt(S^d_sym / S^d) "
        "a_symj / (2 sqrt c).  psf is centred on pixel (npsf // 2, npsf "
        "// 2); everything is zero beyond klim and on the Nyquist row and "
        "column.",
        py::arg("noise_spectrum"), py::arg("psf"), py::arg("klim"),
        py::arg("c") = 0.1, py::arg("delta") = 0.05,
        py::arg("symmetrize") = true, py::arg("noise_rotated") = false,
        py::arg("mode") = "spectral"
    );

    rn.def("total_spectrum",
        [](const RenoiseKernels& k) {
            return half_plane_array(total_spectrum(k), k.n);
        },
        "Total deconvolved noise spectrum of the renoised image: "
        "S^d + S^d_rot in the rotate mode, (1 + c) S^d_sym in the "
        "spectral mode",
        py::arg("kernels")
    );

    rn.def("variance_factor", &variance_factor,
        "sum |h|^2 S^d_total / sum |h|^2 S^d_0 for a Gaussian h of the "
        "given sigma [arcsec]: the ratio of the renoised noise variance "
        "to the white-noise one (2 for white noise in the rotate mode)",
        py::arg("kernels"), py::arg("sigma_arcsec")
    );

    rn.def("gaussian_flux_variance", &gaussian_flux_variance,
        "Variance of the Gaussian aperture flux (anacal.task's "
        "gaussian_flux_variance) under the total noise of the kernels",
        py::arg("kernels"), py::arg("sigma_kernel"), py::arg("sigma_smooth")
    );

    py::class_<Setup>(rn, "Setup")
        .def(py::init([](double c,
                         const std::vector<NoiseSpectrum>& spectra,
                         bool noise_rotated,
                         const std::string& mode) {
                Setup s;
                s.c = c;
                s.spectra = spectra;
                s.noise_rotated = noise_rotated;
                s.mode = mode_from_string(mode);
                return s;
            }),
            "Noise spectra of one patch for the renoising: one "
            "NoiseSpectrum per band on the cell grid -- the raw one "
            "(make_noise_spectrum) for mode 'rotate', the one symmetrized "
            "with the patch PSF for mode 'spectral' (the image carries the "
            "filler) -- the added-noise fraction c (spectral), and whether "
            "the pure-noise image was drawn from the 90-degree-rotated "
            "correlation (xlens convention).  The Task sums the noise "
            "variances over the total spectrum in either mode.",
            py::arg("c") = 0.1,
            py::arg("spectra") = std::vector<NoiseSpectrum>{},
            py::arg("noise_rotated") = true,
            py::arg("mode") = "spectral"
        )
        .def_property_readonly("mode", [](const Setup& s) {
            return std::string(s.mode == Mode::rotate ? "rotate" : "spectral");
        })
        .def_readwrite("c", &Setup::c)
        .def_readwrite("noise_rotated", &Setup::noise_rotated)
        .def_property_readonly("nband", [](const Setup& s) {
            return s.spectra.size();
        });

    rn.def("make_cell_kernels", &make_cell_kernels,
        "One RenoiseKernels per band for the PSF stamp (or band stack) of "
        "a cell",
        py::arg("setup"), py::arg("psf"), py::arg("klim")
    );

    rn.def("rotate90",
        [](const py::array_t<double>& arr) {
            int n = 0;
            const std::vector<double> v = half_plane_vector(arr, n);
            return half_plane_array(rotate90_even(v, n), n);
        },
        "a(R k) = a(-k2, k1) of a real, even half-plane array",
        py::arg("arr")
    );
}

} // namespace renoise
} // namespace anacal
