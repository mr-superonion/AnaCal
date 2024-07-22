#include "anacal.h"


namespace anacal {

FpfsImage::FpfsImage(
    int nx,
    int ny,
    double scale,
    double sigma_arcsec,
    double klim,
    const py::array_t<double>& psf_array,
    bool use_estimate,
    int n_overlap
): cimg(nx, ny, scale, use_estimate), psf_array(psf_array) {
    if ((sigma_arcsec <= 0) || (sigma_arcsec > 5.0)) {
        throw std::runtime_error(
            "FPFS Error: invalid input sigma_arcsec"
        );
    }
    this->nx = nx;
    this->ny = ny;
    this->scale = scale;
    this->sigma_arcsec = sigma_arcsec;
    this->klim = klim;
    this->sigma_f = 1.0 / sigma_arcsec;
    this->fft_ratio = 1.0 / scale / scale;
    if ((n_overlap % 2 != 0) || (n_overlap < 0)) {
        throw std::runtime_error(
            "FPFS Error: n_overlap is not an even number"
        );
    }
    this->n_overlap = n_overlap;
    return;
}


py::array_t<double>
FpfsImage::smooth_image(
    const py::array_t<double>& img_array,
    bool do_rotate,
    int x,
    int y
) {
    const Gaussian gauss_model(sigma_f);
    std::optional<py::array_t<double>> noise_conv;
    // Prepare PSF
    cimg.set_r(psf_array, -1, -1, true);
    cimg.fft();
    if (do_rotate) {
        cimg.rotate90_f();
    }
    const py::array_t<std::complex<double>> parr = cimg.draw_f();
    // signal
    cimg.set_r(img_array, x, y, false);
    cimg.fft();
    // Deconvolve the PSF
    cimg.deconvolve(parr, klim);
    // Convolve Gaussian
    cimg.filter(gauss_model);
    // Back to Real space
    cimg.ifft();
    py::array_t<double> img_conv = cimg.draw_r();
    return img_conv;
}


py::array_t<double>
FpfsImage::smooth_image(
    const py::array_t<double>& gal_array,
    const std::optional<py::array_t<double>>& noise_array,
    int x,
    int y
) {
    //TODO: use different PSF here
    py::array_t<double> gal_conv = this->smooth_image(
        gal_array,
        false,
        x,
        y
    );
    if (noise_array.has_value()) {
        py::array_t<double> noise_conv = this->smooth_image(
            *noise_array,
            true,
            x,
            y
        );
        auto g_r = gal_conv.mutable_unchecked<2>();
        auto n_r = noise_conv.mutable_unchecked<2>();
        for (ssize_t j = 0; j < this->ny; ++j) {
            for (ssize_t i = 0; i < this->nx; ++i) {
                g_r(j, i) = g_r(j, i) + n_r(j, i);
            }
        }
    }
    return gal_conv;
}

void
FpfsImage::find_peak(
    std::vector<std::tuple<int, int, bool>>& peaks,
    const py::array_t<double>& gal_conv,
    double fthres,
    double pthres,
    double std_m00,
    double std_v
) {
    // Never use detections that is too close to boundary
    int bound = std::max(this->n_overlap / 2, 3);
    auto r = gal_conv.unchecked<2>();
    if ((r.shape(0) != this->ny)  || (r.shape(1) != this->nx)) {
        throw std::runtime_error(
            "FPFS Error: convolved image has wrong shape in find_peak."
        );
    }

    double fcut = fthres * std_m00;
    double pcut = fpfs_pnr * std_v;
    double sigma_v = fpfs_cut_sigma_ratio * std_v;

    double wdet_cut = pthres - fpfs_det_sigma2 - 0.02;
    if (std::fabs(wdet_cut) < 1e-10) {
        wdet_cut = 0.0;
    }
    if (wdet_cut < 0.0) {
        throw std::runtime_error(
            "FPFS Error: The second selection threshold pthres is too small."
        );
    }

    for (ssize_t j = bound; j < this->ny - bound; ++j) {
        for (ssize_t i = bound; i < this->nx - bound; ++i) {
            double c = r(j, i);
            double d1 = c - r(j, i+1);
            double d2 = c - r(j+1, i);
            double d3 = c - r(j, i-1);
            double d4 = c - r(j-1, i);
            double s1 = math::ssfunc2(d1, sigma_v - pcut, sigma_v);
            double s2 = math::ssfunc2(d2, sigma_v - pcut, sigma_v);
            double s3 = math::ssfunc2(d3, sigma_v - pcut, sigma_v);
            double s4 = math::ssfunc2(d4, sigma_v - pcut, sigma_v);
            double wdet = s1 * s2 * s3 * s4;
            bool sel = (
                (c > fcut) &&
                (wdet > wdet_cut)
            );
            if (sel) {
                bool is_peak = (
                    (c > r(j-1, i)) &&
                    (c > r(j+1, i)) &&
                    (c > r(j, i-1)) &&
                    (c > r(j, i+1))
                );
                peaks.emplace_back(j, i, is_peak);
            }
        }
    }
    return;
}


py::array_t<FpfsPeaks>
FpfsImage::detect_source(
    py::array_t<double>& gal_array,
    double fthres,
    double pthres,
    double std_m00,
    double std_v,
    int bound,
    const std::optional<py::array_t<double>>& noise_array,
    const std::optional<py::array_t<int16_t>>& mask_array
) {

    std::vector<std::tuple<int, int, bool>> peaks;
    auto r = gal_array.unchecked<2>();
    // Determine number of patches
    // y direction
    int npix_y = r.shape(0);
    int npatch_y = npix_y / (this->ny - this->n_overlap);
    float npatch_y_f = npix_y / (this->ny - this->n_overlap);
    if (npatch_y > npatch_y_f) {
        npatch_y = npatch_y + 1;
    }
    int npix2_y = npatch_y * (this->ny - this->n_overlap) + this->n_overlap;
    int npad_y = (npix2_y - npix_y) / 2;

    // x direction
    int npix_x = r.shape(1);
    int npatch_x = npix_x / (this->nx - this->n_overlap);
    float npatch_x_f = npix_x / (this->nx - this->n_overlap);
    if (npatch_x > npatch_x_f) {
        npatch_x = npatch_x + 1;
    }
    int npix2_x = npatch_x * (this->nx - this->n_overlap) + this->n_overlap ;
    int npad_x = (npix2_x - npix_x) / 2;

    // Do detection in each patch
    for (int j = 0; j < npatch_y; ++j) {
        int yc = (this->ny - this->n_overlap) * j + this->ny / 2 - npad_y;
        int ymin = (this->ny - this->n_overlap) * j - npad_y;
        for (int i = 0; i < npatch_x; ++i) {
            int xc = (this->nx - this->n_overlap) * i + this->nx / 2 - npad_x;
            int xmin = (this->nx - this->n_overlap) * i - npad_x;
            py::array_t<double> gal_conv = this->smooth_image(
                gal_array,
                noise_array,
                xc,
                yc
            );

            this->find_peak(
                peaks,
                gal_conv,
                fthres,
                pthres,
                std_m00,
                std_v
            );
            int npeaks = peaks.size();
            for (int i = 0; i < npeaks;) {
                int y = std::get<0>(peaks[i]) + ymin;
                int x = std::get<1>(peaks[i]) + xmin;
                std::get<0>(peaks[i]) = y;
                std::get<1>(peaks[i]) = x;
                bool cond = (
                    (y > bound) && (y < npix_y - bound) &&
                    (x > bound) && (x < npix_x - bound)
                );
                if (cond) {
                    ++i;
                } else {
                    peaks.erase(peaks.begin() + i);
                    --npeaks;
                }
            }
        }
    }

    int nrow = peaks.size();
    py::array_t<FpfsPeaks> detection(nrow);
    auto src_r = detection.mutable_unchecked<1>();

    for (ssize_t j = 0; j < nrow; ++j) {
        const auto& elem = peaks[j];
        src_r(j).y = std::get<0>(elem);
        src_r(j).x = std::get<1>(elem);
        src_r(j).is_peak = int(std::get<2>(elem));
        src_r(j).mask_value = 0;
    }

    if (mask_array.has_value()) {
        add_pixel_mask_column(
            detection,
            *mask_array,
            sigma_arcsec,
            scale
        );
    }
    return detection;
}

py::array_t<double>
FpfsImage::measure_source(
    const py::array_t<double>& gal_array,
    const py::array_t<std::complex<double>>& filter_image,
    const std::optional<py::array_t<double>>& psf_array,
    const std::optional<py::array_t<FpfsPeaks>>& det,
    bool do_rotate
) {
    ssize_t ndim = filter_image.ndim();
    if ( ndim != 3) {
        throw std::runtime_error(
            "FPFS Error: Input filter image must be 3-dimensional."
        );
    }


    const py::array_t<double>&
        psf_use = psf_array.has_value() ? *psf_array : this->psf_array;
    cimg.set_r(psf_use, -1, -1, false);
    cimg.fft();
    if (do_rotate){
        cimg.rotate90_f();
    }
    const py::array_t<std::complex<double>> parr = cimg.draw_f();
    const py::array_t<std::complex<double>> fimg = deconvolve_filter(
        filter_image,
        parr,
        scale,
        klim
    );

    ssize_t ncol = filter_image.shape()[ndim - 1];
    py::array_t<FpfsPeaks> det_default(1);
    auto r = det_default.mutable_unchecked<1>();
    r(0).y = ny / 2; r(0).x = nx / 2;
    r(0).is_peak = 1; r(0).mask_value = 0;
    const py::array_t<FpfsPeaks>& det_use = det.has_value() ? *det : det_default;
    auto det_r = det_use.unchecked<1>();

    ssize_t nrow = det_use.shape()[0];
    py::array_t<double> src({nrow, ncol});
    auto src_r = src.mutable_unchecked<2>();
    for (ssize_t j = 0; j < nrow; ++j) {
        int y = det_r(j).y; int x = det_r(j).x;
        cimg.set_r(gal_array, x, y, false);
        cimg.fft();
        py::array_t<double> row = cimg.measure(fimg);
        auto row_r = row.unchecked<1>();
        for (ssize_t i = 0; i < ncol; ++i) {
            src_r(j, i) = row_r(i) * fft_ratio;
        }
    }
    return src;
}

py::array_t<double>
FpfsImage::measure_source(
    const py::array_t<double>& gal_array,
    const py::array_t<std::complex<double>>& filter_image,
    const BasePsf& psf_obj,
    const std::optional<py::array_t<FpfsPeaks>>& det,
    bool do_rotate
) {
    ssize_t ndim = filter_image.ndim();
    if ( ndim != 3) {
        throw std::runtime_error(
            "FPFS Error: Input filter image must be 3-dimensional."
        );
    }
    ssize_t ncol = filter_image.shape()[ndim - 1];

    py::array_t<FpfsPeaks> det_default(1);
    auto r = det_default.mutable_unchecked<1>();
    r(0).y = ny / 2; r(0).x = nx / 2;
    r(0).is_peak = 1; r(0).mask_value = 0;
    const py::array_t<FpfsPeaks>& det_use = det.has_value() ? *det : det_default;
    auto det_r = det_use.unchecked<1>();

    ssize_t nrow = det_use.shape()[0];
    py::array_t<double> src({nrow, ncol});
    auto src_r = src.mutable_unchecked<2>();
    for (ssize_t j = 0; j < nrow; ++j) {
        int y = det_r(j).y; int x = det_r(j).x;
        {
            py::array_t<double> psf_use = psf_obj.draw(x, y);
            cimg.set_r(psf_use, -1, -1, false);
        }
        cimg.fft();
        if (do_rotate){
            cimg.rotate90_f();
        }
        {
            const py::array_t<std::complex<double>> parr = cimg.draw_f();
            cimg.set_r(gal_array, x, y, false);
            cimg.fft();
            cimg.deconvolve(parr, klim);
        }
        py::array_t<double> row = cimg.measure(filter_image);
        auto row_r = row.unchecked<1>();
        for (ssize_t i = 0; i < ncol; ++i) {
            src_r(j, i) = row_r(i) * fft_ratio;
        }
    }
    return src;
}


FpfsImage::~FpfsImage() {
}


void
pyExportFpfs(py::module& m) {
    py::module_ fpfs = m.def_submodule("fpfs", "submodule for FPFS shear estimation");
    fpfs.attr("fpfs_cut_sigma_ratio") = fpfs_cut_sigma_ratio;
    fpfs.attr("fpfs_det_sigma2") = fpfs_det_sigma2;
    fpfs.attr("fpfs_pnr") = fpfs_pnr;
    py::class_<FpfsImage>(fpfs, "FpfsImage")
        .def(py::init<
                int, int, double, double, double, const py::array_t<double>&,
                bool, int
            >(),
            "Initialize the FpfsImage object using an ndarray",
            py::arg("nx"), py::arg("ny"),
            py::arg("scale"),
            py::arg("sigma_arcsec"),
            py::arg("klim"),
            py::arg("psf_array"),
            py::arg("use_estimate")=true,
            py::arg("n_overlap")=0
        )
        .def("smooth_image",
            py::overload_cast<
                const py::array_t<double>&,
                bool,
                int,
                int
            >(&FpfsImage::smooth_image),
            "Smooths the image after PSF deconvolution",
            py::arg("gal_array"),
            py::arg("do_rotate")=false,
            py::arg("x")=-1,
            py::arg("y")=-1
        )
        .def("smooth_image",
            py::overload_cast<
                const py::array_t<double>&,
                const std::optional<py::array_t<double>>&,
                int,
                int
            >(&FpfsImage::smooth_image),
            "Smooths the image after PSF deconvolution",
            py::arg("gal_array"),
            py::arg("noise_array")=py::none(),
            py::arg("x")=-1,
            py::arg("y")=-1
        )
        .def("detect_source",
            &FpfsImage::detect_source,
            "Detect galaxy candidates from image",
            py::arg("gal_array"),
            py::arg("fthres"),
            py::arg("pthres"),
            py::arg("std_m00"),
            py::arg("std_v"),
            py::arg("bound"),
            py::arg("noise_array")=py::none(),
            py::arg("mask_array")=py::none()
        )
        .def("measure_source",
            py::overload_cast<
                const py::array_t<double>&,
                const py::array_t<std::complex<double>>&,
                const std::optional<py::array_t<double>>&,
                const std::optional<py::array_t<FpfsPeaks>>&,
                bool
            >(&FpfsImage::measure_source),
            "measure source properties using filter at the position of det",
            py::arg("gal_array"),
            py::arg("filter_image"),
            py::arg("psf_array")=py::none(),
            py::arg("det")=py::none(),
            py::arg("do_rotate")=false
        )
        .def("measure_source",
            py::overload_cast<
                const py::array_t<double>&,
                const py::array_t<std::complex<double>>&,
                const BasePsf&,
                const std::optional<py::array_t<FpfsPeaks>>&,
                bool
            >(&FpfsImage::measure_source),
            "measure source properties using filter at the position of det",
            py::arg("gal_array"),
            py::arg("filter_image"),
            py::arg("psf_obj"),
            py::arg("det")=py::none(),
            py::arg("do_rotate")=false
        );
}
}
