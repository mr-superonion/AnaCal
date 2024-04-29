#include "anacal.h"


namespace anacal {
    py::array_t<double>
    simulate_noise(
        unsigned int seed,
        double noise_std,
        int nx,
        int ny
    ) {

        std::mt19937 engine(seed);
        std::normal_distribution<double> dist(0.0, noise_std);

        auto result = py::array_t<double>({ny, nx});
        auto r = result.mutable_unchecked<2>();

        for (ssize_t j = 0; j < ny; ++j) {
            for (ssize_t i = 0; i < nx; ++i) {
                r(j, i) = dist(engine);
            }
        }
        return result;
    }


    py::array_t<double>
    simulate_noise(
        unsigned int seed,
        const py::array_t<double>& correlation,
        int nx,
        int ny,
        double scale,
        bool do_rotate=false
    ) {
        Image image(nx, ny, scale);
        image.set_noise_f(seed, correlation);
        if (do_rotate){
            image.rotate90_f();
        }
        image.ifft();
        py::array_t<double> result = image.draw_r();
        return result;
    }


    py::array_t<double>
    simulate_noise_correlation(
        double noise_std,
        const BaseModel& corr_model,
        int nx,
        int ny,
        double scale
    ) {
        Image image(nx, ny, scale);
        image.set_delta_f();
        image.filter(corr_model);
        image.ifft();
        // shift to center
        py::array_t<double> result = image.draw_r(true);
        auto r = result.mutable_unchecked<2>();
        double ratio = noise_std * noise_std / r(ny / 2, nx / 2);
        ssize_t nx2 = nx / 2;
        ssize_t ny2 = ny / 2;
        ssize_t rcut = std::min(nx2, ny2) - 2;
        ssize_t rcut2 = rcut * rcut;
        for (ssize_t j = 0; j < ny; ++j) {
            ssize_t y = j - ny2;
            for (ssize_t i = 0; i < nx; ++i) {
                ssize_t x = i - nx2;
                ssize_t rr = x * x + y * y;
                if (rr < rcut2) {
                    r(j, i) = r(j, i) * ratio;
                } else {
                    r(j, i) = 0.0;
                }

            }
        }
        return result;
    }


    void
    pyExportNoise(py::module& m) {
        py::module_ noise = m.def_submodule(
            "noise", "submodule for noise simulation"
        );
        noise.def("simulate_noise",
            py::overload_cast<unsigned int, double, int, int>
                (&simulate_noise),
            "simulate noise in configuration space",
            py::arg("seed"),
            py::arg("noise_std"),
            py::arg("nx"),
            py::arg("ny")
        );
        noise.def("simulate_noise",
            py::overload_cast
            <unsigned int, const py::array_t<double>&, int, int, double, bool>
                (&simulate_noise),
            "simulate noise in configuration space",
            py::arg("seed"),
            py::arg("correlation"),
            py::arg("nx"),
            py::arg("ny"),
            py::arg("scale"),
            py::arg("do_rotate")=false
        );
        noise.def("simulate_noise_correlation",
            py::overload_cast
            <double, const BaseModel&, int, int, double>
                (&simulate_noise_correlation),
            "simulate noise in configuration space",
            py::arg("noise_std"),
            py::arg("corr_model"),
            py::arg("nx"),
            py::arg("ny"),
            py::arg("scale")
        );
    }
}
