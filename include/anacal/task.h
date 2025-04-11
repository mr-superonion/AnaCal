#ifndef ANACAL_TASK_H
#define ANACAL_TASK_H

#include "detector.h"
#include "stdafx.h"
#include "mask2.h"

namespace anacal {
namespace task {



inline double get_smoothed_variance(
    double scale,
    double sigma_arcsec_det,
    const py::array_t<double>& psf_array,
    double variance
) {
    double variance_sm = 0.0;
    // number of pixels in x and y used to estimated noise variance
    // result is independent on this
    int npix = 64;
    Image img_obj(npix, npix, scale, true);
    {
        // Prepare PSF
        img_obj.set_r(psf_array, -1, -1, true);
        img_obj.fft();
        const py::array_t<std::complex<double>> parr = img_obj.draw_f();
        {
            // white noise
            auto pf = py::array_t<std::complex<double>>({npix, npix / 2 + 1});
            auto pf_r = pf.mutable_unchecked<2>();
            std::complex<double> vv(std::sqrt(variance) / npix, 0.0);
            std::complex<double> sqrt2vv = std::sqrt(2.0) * vv;
            for (ssize_t j = 0; j < npix; ++j) {
                for (ssize_t i = 1; i < npix / 2; ++i) {
                    pf_r(j, i) = sqrt2vv;
                }
                pf_r(j, 0) = vv;
                pf_r(j, npix / 2) = vv;
            }
            img_obj.set_f(pf);
        }
        // Deconvolve the PSF
        img_obj.deconvolve(parr, 3.0 / scale);
    }
    {
        // Convolve Gaussian
        const Gaussian gauss_model(1.0 / sigma_arcsec_det);
        img_obj.filter(gauss_model);
    }
    {
        const py::array_t<std::complex<double>> pf_dec = img_obj.draw_f();
        auto pfd_r = pf_dec.unchecked<2>();
        for (ssize_t j = 0; j < npix; ++j) {
            for (ssize_t i = 0; i < npix / 2 + 1; ++i) {
                variance_sm += std::norm(pfd_r(j, i));
            }
        }
    }
    return variance_sm;
};


class TaskAlpha {
public:
    // stamp dimension
    double scale;
    double sigma_arcsec_det;
    double sigma_arcsec;
    double snr_peak_min, omega_f, v_min, omega_v, p_min, omega_p;
    const ngmix::modelPrior prior;
    int stamp_size, ss2;
    int image_bound;
    int num_epochs;
    bool force_size, force_center;
    ngmix::GaussFit fitter;

    TaskAlpha(
        double scale,
        double sigma_arcsec_det,
        double sigma_arcsec,
        double snr_peak_min,
        double omega_f,
        double v_min,
        double omega_v,
        double p_min,
        double omega_p,
        const std::optional<ngmix::modelPrior>& prior=std::nullopt,
        int stamp_size=64,
        int image_bound=0,
        int num_epochs=10,
        bool force_size=false,
        bool force_center=false,
        double fpfs_c0=1.0
    ) : scale(scale), sigma_arcsec_det(sigma_arcsec_det),
        sigma_arcsec(sigma_arcsec), snr_peak_min(snr_peak_min),
        omega_f(omega_f), v_min(v_min), omega_v(omega_v),
        p_min(p_min), omega_p(omega_p),
        prior(prior ? *prior : ngmix::modelPrior()),
        stamp_size(stamp_size), image_bound(image_bound),
        num_epochs(num_epochs), fitter(
            scale, sigma_arcsec, stamp_size,
            force_size, force_center,
            fpfs_c0
        )
    {
        if (stamp_size % 2 != 0 ) {
            throw std::invalid_argument("nx or ny is not even number");
        }
        if (sigma_arcsec_det <= 0) {
            throw std::invalid_argument("sigma_arcsec_det must be positive");
        }
        if (sigma_arcsec <= 0) {
            throw std::invalid_argument("sigma_arcsec must be positive");
        }
    };

    inline std::vector<table::galNumber>
    process_block_impl(
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        double variance_det,
        double variance_meas,
        const geometry::block & block,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt
    ) {

        std::vector<table::galNumber> catalog = detector::find_peaks(
            img_array,
            psf_array,
            this->sigma_arcsec_det,
            this->snr_peak_min * std::pow(variance_det, 0.5),
            this->omega_f,
            this->v_min,
            this->omega_v,
            this->p_min,
            this->omega_p,
            block,
            noise_array,
            this->image_bound
        );
        return this->fitter.process_block(
            catalog,
            img_array,
            psf_array,
            this->prior,
            noise_array,
            this->num_epochs,
            variance_meas,
            block
        );
    };

    inline
    void select_push_back(
        table::galNumber& src,
        std::vector<table::galNumber> & catalog,
        double std_fpfs
    ) {
        src.wdet = src.wdet * math::ssfunc1(
            src.fpfs_m0,
            6.0 * std_fpfs,
            std_fpfs
        );
        src.wdet = src.wdet * math::ssfunc1(
            src.fpfs_m2 - 0.09 * src.fpfs_m0,
            std_fpfs,
            std_fpfs
        );
        if (src.wdet.v > 1e-8) catalog.push_back(src);
        return;
    };

    inline py::array_t<table::galRow>
    process_image(
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        double variance,
        const std::optional<std::vector<geometry::block>>& block_list=std::nullopt,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt,
        const std::optional<py::array_t<int16_t>>& mask_array=std::nullopt
    ) {

        int image_ny = img_array.shape(0);
        int image_nx = img_array.shape(1);

        std::vector<geometry::block> bb_list = block_list ? *block_list
            : geometry::get_block_list(
                image_nx, image_ny, image_nx, image_ny, 0, this->scale
            );

        std::vector<table::galNumber> catalog;
        for (const geometry::block & block: bb_list) {
            double variance_use;
            if (noise_array.has_value()) {
                variance_use = variance * 2.0;
            } else {
                variance_use = variance;
            }
            double variance_det = get_smoothed_variance(
                block.scale,
                sigma_arcsec_det,
                psf_array,
                variance_use
            );
            double variance_meas = get_smoothed_variance(
                block.scale,
                sigma_arcsec,
                psf_array,
                variance_use
            );
            double std_fpfs = std::pow(
                get_smoothed_variance(
                    block.scale,
                    sigma_arcsec * 1.414,
                    psf_array,
                    variance_use
                ),
                0.5
            );
            std::vector<table::galNumber> catb = process_block_impl(
                img_array,
                psf_array,
                variance_det,
                variance_meas,
                block,
                noise_array
            );
            for (const table::galNumber& src : catb) {
                table::galNumber src_new = src.decentralize(block);
                this->select_push_back(src_new, catalog, std_fpfs);
            }
        }

        if (mask_array.has_value()) {
            mask2::add_pixel_mask_column_catalog(
                catalog,
                *mask_array,
                sigma_arcsec,
                scale
            );
        }
        return table::objlist_to_array(catalog);
    };
};

void pyExportTask(py::module_& m);

} // task
} // anacal

#endif // ANACAL_TASK_H
