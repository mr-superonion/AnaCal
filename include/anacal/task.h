#ifndef ANACAL_TASK_H
#define ANACAL_TASK_H

#include "detector.h"
#include "stdafx.h"

namespace anacal {
namespace task {

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
    bool force_size, force_shape, force_center;
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
        bool force_shape=false,
        bool force_center=false
    ) : scale(scale), sigma_arcsec_det(sigma_arcsec_det),
        sigma_arcsec(sigma_arcsec), snr_peak_min(snr_peak_min),
        omega_f(omega_f), v_min(v_min), omega_v(omega_v),
        p_min(p_min), omega_p(omega_p),
        prior(prior ? *prior : ngmix::modelPrior()),
        stamp_size(stamp_size), image_bound(image_bound),
        num_epochs(num_epochs), fitter(
            scale, sigma_arcsec, stamp_size,
            force_size, force_shape, force_center
        )
    {
        if (stamp_size % 2 != 0 ) {
            throw std::invalid_argument("nx or ny is not even number");
        }
        if (sigma_arcsec <= 0) {
            throw std::invalid_argument("sigma_arcsec must be positive");
        }
    };

    inline std::vector<table::galNumber>
    process_block_impl(
        py::array_t<double>& img_array,
        py::array_t<double>& psf_array,
        double variance,
        const geometry::block & block,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt
    ) {
        double noise_std = std::pow(variance, 0.5);
        std::vector<table::galNumber> catalog = detector::find_peaks(
            img_array,
            psf_array,
            this->sigma_arcsec_det,
            this->snr_peak_min * noise_std,
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
            variance,
            block
        );
    };

    inline std::vector<table::galNumber>
    process_image_impl(
        py::array_t<double>& img_array,
        py::array_t<double>& psf_array,
        double variance,
        const std::vector<geometry::block>& block_list,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt
    ) {
        std::vector<table::galNumber> catalog;
        for (const geometry::block & block: block_list) {
            std::vector<table::galNumber> v = process_block_impl(
                img_array,
                psf_array,
                variance,
                block,
                noise_array
            );
            for (const auto& element : v) {
                catalog.push_back(element.decentralize(block));
            }
        }
        return catalog;
    };

    inline py::array_t<table::galRow>
    process_image(
        py::array_t<double>& img_array,
        py::array_t<double>& psf_array,
        double variance,
        const std::optional<std::vector<geometry::block>>& block_list=std::nullopt,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt
    ) {

        int image_ny = img_array.shape(0);
        int image_nx = img_array.shape(1);

        std::vector<geometry::block> bb_list = block_list ? *block_list
            : geometry::get_block_list(
                image_nx, image_ny, image_nx, image_ny, 0, this->scale
            );
        std::vector<table::galNumber> catalog = this->process_image_impl(
            img_array,
            psf_array,
            variance,
            bb_list,
            noise_array
        );
        return table::objlist_to_array(catalog);
    };
};

void pyExportTask(py::module_& m);

} // task
} // anacal

#endif // ANACAL_TASK_H
