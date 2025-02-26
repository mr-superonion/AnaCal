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
    double snr_peak_min, omega_f, v_min, omega_v, pthres;
    ngmix::modelPrior prior;
    int stamp_size, ss2;
    int image_bound;
    int num_epochs;

    TaskAlpha(
        double scale,
        double sigma_arcsec_det,
        double sigma_arcsec,
        double snr_peak_min,
        double omega_f,
        double v_min,
        double omega_v,
        double pthres,
        const ngmix::modelPrior & prior,
        int stamp_size=64,
        int image_bound=0,
        int num_epochs=10
    ) : scale(scale), sigma_arcsec_det(sigma_arcsec_det),
        sigma_arcsec(sigma_arcsec), snr_peak_min(snr_peak_min),
        omega_f(omega_f), v_min(v_min), omega_v(omega_v), pthres(pthres),
        prior(prior), stamp_size(stamp_size), image_bound(image_bound),
        num_epochs(num_epochs)
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
            this->pthres,
            block,
            noise_array,
            this->image_bound
        );
        ngmix::GaussFit fitter(
            this->scale,
            this->sigma_arcsec,
            this->stamp_size
        );
        return fitter.process_block(
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
        std::vector<table::galNumber> catalog;
        for (const geometry::block & block: bb_list) {
            std::vector<table::galNumber> v=process_block_impl(
                img_array,
                psf_array,
                variance,
                block,
                noise_array
            );
            catalog.insert(catalog.end(), v.begin(), v.end());
        }
        return table::objlist_to_array(catalog);
    };
};

void pyExportTask(py::module_& m);

} // task
} // anacal

#endif // ANACAL_TASK_H
