#ifndef ANACAL_TASK_H
#define ANACAL_TASK_H

#include "detector.h"
#include "stdafx.h"
#include "mask2.h"
#include "psf.h"

namespace anacal {
namespace task {


class TaskAlpha {
public:
    // stamp dimension
    double scale;
    double sigma_arcsec;
    double snr_peak_min, omega_f, v_min, omega_v, p_min, omega_p;
    const ngmix::modelPrior prior;
    int stamp_size, ss2;
    int image_bound;
    int num_epochs;
    bool force_size, force_center;
    ngmix::GaussFit fitter;
    double sigma_arcsec_det;

    TaskAlpha(
        double scale,
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
    ) : scale(scale), sigma_arcsec(sigma_arcsec), snr_peak_min(snr_peak_min),
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
        if (sigma_arcsec <= 0) {
            throw std::invalid_argument("sigma_arcsec must be positive");
        }
        this->sigma_arcsec_det = sigma_arcsec * 1.414;
    };

    inline std::vector<table::galNumber>
    process_block(
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        double variance,
        const geometry::block & block,
        const std::optional<py::array_t<table::galRow>>& detection=std::nullopt,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt
    ) {
        std::vector<table::galNumber> catalog;
        if (detection.has_value()) {
            catalog = table::array_to_objlist(
                *detection,
                block
            );
        } else {
            catalog = detector::find_peaks(
                img_array,
                psf_array,
                this->sigma_arcsec,
                this->snr_peak_min,
                variance,
                this->omega_f,
                this->v_min,
                this->omega_v,
                this->p_min,
                this->omega_p,
                block,
                noise_array,
                this->image_bound
            );
        }
        this->fitter.process_block_impl(
            catalog,
            img_array,
            psf_array,
            this->prior,
            this->num_epochs,
            variance,
            block,
            noise_array
        );
        return catalog;
    };

    inline py::array_t<table::galRow>
    process_image(
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        double variance,
        const std::vector<geometry::block>& block_list,
        const std::optional<py::array_t<table::galRow>>& detection=std::nullopt,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt,
        const std::optional<py::array_t<int16_t>>& mask_array=std::nullopt
    ) {

        std::vector<table::galNumber> catalog;
        for (const geometry::block & block: block_list) {
            double variance_use;
            if (noise_array.has_value()) {
                variance_use = variance * 2.0;
            } else {
                variance_use = variance;
            }
            std::vector<table::galNumber> catb = process_block(
                img_array,
                psf_array,
                variance_use,
                block,
                detection,
                noise_array
            );
            for (const table::galNumber& src : catb) {
                catalog.push_back(src.decentralize(block));
            }
        }

        if (mask_array.has_value()) {
            mask2::add_pixel_mask_column_catalog(
                catalog,
                *mask_array,
                this->sigma_arcsec_det,
                scale
            );
        }
        return table::objlist_to_array(catalog);
    };

    inline py::array_t<table::galRow>
    process_image(
        const py::array_t<double>& img_array,
        const psf::BasePsf& psf_obj,
        double variance,
        const std::vector<geometry::block>& block_list,
        const std::optional<py::array_t<table::galRow>>& detection=std::nullopt,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt,
        const std::optional<py::array_t<int16_t>>& mask_array=std::nullopt
    ) {

        std::vector<table::galNumber> catalog;
        for (const geometry::block & block: block_list) {
            double variance_use;
            if (noise_array.has_value()) {
                variance_use = variance * 2.0;
            } else {
                variance_use = variance;
            }
            py::array_t<double> psf_array = psf_obj.draw(
                block.xcen, block.ycen
            );
            std::vector<table::galNumber> catb = process_block(
                img_array,
                psf_array,
                variance_use,
                block,
                detection,
                noise_array
            );
            for (const table::galNumber& src : catb) {
                catalog.push_back(src.decentralize(block));
            }
        }

        if (mask_array.has_value()) {
            mask2::add_pixel_mask_column_catalog(
                catalog,
                *mask_array,
                this->sigma_arcsec_det,
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
