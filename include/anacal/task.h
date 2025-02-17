#ifndef ANACAL_TASK
#define ANACAL_TASK

#include "anacal.h"

namespace anacal {
namespace task {

Task::Task(
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
            "Task Error: invalid input sigma_arcsec"
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
            "Task Error: npix_overlap is not an even number"
        );
    }
    this->npix_overlap = npix_overlap;
    this->bound = bound;
    return;
}


void
Task::find_peaks(
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


py::array_t<FpfsPeaks>
Task::detect_source(
    py::array_t<double>& gal_array,
    double fthres,
    double pthres,
    double std_m00,
    double v_min,
    double omega_v,
    const std::optional<py::array_t<double>>& noise_array,
    const std::optional<py::array_t<int16_t>>& mask_array
) {

    auto r = gal_array.unchecked<2>();
    this->ny_array = r.shape(0);
    this->nx_array = r.shape(1);

    // Determine number of patches
    // y direction
    int npatch_y = this->ny_array / (this->ny - this->npix_overlap);
    float npatch_y_f = this->ny_array / (this->ny - this->npix_overlap + 0.0);
    if (npatch_y_f > npatch_y) {
        npatch_y = npatch_y + 1;
    }
    int ny2 = npatch_y * (this->ny - this->npix_overlap) + this->npix_overlap;
    int npad_y = (ny2 - this->ny_array) / 2;

    // x direction
    int npatch_x = this->nx_array / (this->nx - this->npix_overlap);
    float npatch_x_f = this->nx_array / (this->nx - this->npix_overlap + 0.0);
    if (npatch_x_f > npatch_x) {
        npatch_x = npatch_x + 1;
    }
    int nx2 = npatch_x * (this->nx - this->npix_overlap) + this->npix_overlap;
    int npad_x = (nx2 - this->nx_array) / 2;

    std::vector<std::tuple<int, int, bool>> peaks;
    // Do detection in each patch
    for (int j = 0; j < npatch_y; ++j) {
        int ycen = (this->ny - this->npix_overlap) * j + this->ny2 - npad_y;
        for (int i = 0; i < npatch_x; ++i) {
            int xcen = (this->nx - this->npix_overlap) * i + this->nx2 - npad_x;
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
        detection = add_pixel_mask_column(
            detection,
            *mask_array,
            sigma_arcsec,
            scale
        );
    }
    return detection;
}

} // task
} // anacal
#endif // ANACAL_TASK
