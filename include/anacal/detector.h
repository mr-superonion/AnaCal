#ifndef ANACAL_DETECTOR
#define ANACAL_DETECTOR

#include "anacal.h"

namespace anacal {

struct detPeak {
    double y=0.0;
    double x=0.0;
    int mask_value=0;
    bool down_weight=false;
    math::qnumber wdet;
};

Detector::Detector(
    int nx,
    int ny,
    double scale,
    double sigma_arcsec,
    double klim,
    const py::array_t<double>& psf_array,
    bool use_estimate,
    int npix_overlap,
    int bound
): img_obj(nx, ny, scale, use_estimate), psf_array(psf_array) {
    if ((sigma_arcsec <= 0) || (sigma_arcsec > 5.0)) {
        throw std::runtime_error(
            "Detector Error: invalid input sigma_arcsec"
        );
    }
    this->nx = nx;
    this->ny = ny;
    this->nx2 = nx / 2;
    this->ny2 = ny / 2;
    this->scale = scale;
    this->sigma_arcsec = sigma_arcsec;
    this->klim = klim;
    this->sigma_f = 1.0 / sigma_arcsec;
    if ((npix_overlap % 2 != 0) || (npix_overlap < 0)) {
        throw std::runtime_error(
            "Detector Error: npix_overlap is not an even number"
        );
    }
    this->npix_overlap = npix_overlap;
    this->bound = bound;
    return;
}


void
Detector::find_peaks(
    std::vector<detPeak>>& peaks,
    const py::array_t<double>& gal_conv,
    double fthres,
    double pthres,
    double std_m00,
    double v_min,
    double omega_v,
    int xcen,
    int ycen
) {
    // Do not use detections that is too close to boundary
    int bound_patch = std::max(this->npix_overlap / 2, 3);
    auto r = gal_conv.unchecked<2>();
    if ((r.shape(0) != this->ny)  || (r.shape(1) != this->nx)) {
        throw std::runtime_error(
            "FPFS Error: convolved image has wrong shape in find_peaks."
        );
    }

    int ymin = ycen - this->ny2;
    int xmin = xcen - this->nx2;

    double fcut = fthres * std_m00;
    double wdet_cut = pthres - fpfs_det_sigma2 - 0.02;

    if (std::fabs(wdet_cut) < 1e-10) {
        wdet_cut = 0.0;
    }
    if (wdet_cut < 0.0) {
        throw std::runtime_error(
            "FPFS Error: The second selection threshold pthres is too small."
        );
    }
    int drmax2 = 1;

    for (int j = bound_patch; j < this->ny - bound_patch; ++j) {
        for (int i = bound_patch; i < this->nx - bound_patch; ++i) {
            double wdet = 1.0;
            double c = r(j, i);
            for (int dj = -1; dj <= 1; dj++) {
                int dj2 = dj * dj;
                for (int di = -1; di <= 1; di++) {
                    int dr2 = di * di + dj2;
                    if ((dr2 <= drmax2) && (dr2 != 0)) {
                        double zv = math::ssfunc2(
                            c - r(j + dj, i + di),
                            v_min,
                            omega_v
                        );
                        wdet = wdet * zv;
                    }
                }
            }
            int y = j + ymin;
            int x = i + xmin;
            bool sel = (
                (c > fcut) &&
                (wdet > wdet_cut) &&
                (y > this->bound) && (y < this->ny_array - this->bound) &&
                (x > this->bound) && (x < this->nx_array - this->bound)
            );
            if (sel) {
                bool is_peak = (
                    (c > r(j-1, i)) &&
                    (c > r(j+1, i)) &&
                    (c > r(j, i-1)) &&
                    (c > r(j, i+1))
                );
                peaks.emplace_back(y, x, is_peak);
            }
        }
    }
    return;
}


std::vector<detPeak>
Detector::detect_source(
    py::array_t<double>& gal_array,
    double fthres,
    double pthres,
    double v_min,
    double omega_v,
    const std::optional<py::array_t<double>>& noise_array
) {

    auto r = gal_array.unchecked<2>();
    this->ny_array = r.shape(0);
    this->nx_array = r.shape(1);

    py::array_t<double> gal_conv = this->smooth_image(
        gal_array,
        noise_array,
        xcen,
        ycen
    );

    this->find_peaks(
        peaks,
        gal_conv,
        fthres,
        pthres,
        std_m00,
        v_min,
        omega_v,
        xcen,
        ycen
    );
    return detection;
}


}

#endif // ANACAL_DETECTOR
