#ifndef ANACAL_TASK_H
#define ANACAL_TASK_H

#include "detector.h"
#include "mask.h"
#include "psf.h"

namespace anacal {
namespace task {


class Task {
public:
    // stamp dimension
    double scale;
    double sigma_arcsec;
    double snr_peak_min, omega_f, v_min, omega_v, p_min, omega_p;
    const ngmix::modelPrior prior;
    int stamp_size, ss2;
    int image_bound;
    int num_epochs;
    int num_epochs_deblend;
    bool force_size, force_center;
    ngmix::GaussFit fitter;
    double sigma_arcsec_det;

    Task(
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
        int num_epochs=3,
        int num_epochs_deblend=3,
        bool force_size=false,
        bool force_center=false,
        double fpfs_c0=1.0
    ) : scale(scale), sigma_arcsec(sigma_arcsec), snr_peak_min(snr_peak_min),
        omega_f(omega_f), v_min(v_min), omega_v(omega_v),
        p_min(p_min), omega_p(omega_p),
        prior(prior ? *prior : ngmix::modelPrior()),
        stamp_size(stamp_size), image_bound(image_bound),
        num_epochs(num_epochs), num_epochs_deblend(num_epochs_deblend), fitter(
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

    inline void
    prepare_indices(
        std::vector<table::galNumber>& catalog,
        geometry::block & block
    ) {
        double x_min = block.xmin * block.scale;
        double y_min = block.ymin * block.scale;
        double x_max = block.xmax * block.scale;
        double y_max = block.ymax * block.scale;
        std::size_t nrow = catalog.size();
        std::vector<std::size_t> indices;
        indices.reserve(static_cast<std::size_t>(nrow / 4));
        for (std::size_t i = 0; i < nrow; ++i) {
            const table::galNumber & src = catalog[i];
            if ((src.x1_det >= x_min) &&
                (src.x1_det < x_max) &&
                (src.x2_det >= y_min) &&
                (src.x2_det < y_max)
            ) {
                indices.push_back(i);
            }
        }
        block.indices = indices;
        return;
    };

    inline std::vector<table::galNumber>
    detect_block(
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        double variance,
        const geometry::block & block,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt
    ) {
        std::vector<table::galNumber> catalog = detector::find_peaks(
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
        for (table::galNumber& src : catalog) {
            src = src.decentralize(block);
        }
        return catalog;
    };

    inline void
    measure_block(
        std::vector<table::galNumber>& catalog,
        std::vector<table::galNumber>& catalog_model,
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        double variance,
        const geometry::block & block,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt,
        int run_id=0
    ) {
        if (block.indices.empty()) return;
        for (std::size_t idx : block.indices) {
            catalog[idx] = catalog[idx].centralize(block);
            catalog_model[idx] = catalog_model[idx].centralize(block);
        }
        this->fitter.process_block_impl(
            catalog,
            catalog_model,
            img_array,
            psf_array,
            this->prior,
            this->num_epochs,
            variance,
            block,
            noise_array,
            run_id
        );
        for (std::size_t idx : block.indices) {
            catalog[idx] = catalog[idx].decentralize(block);
            catalog_model[idx] = catalog_model[idx].decentralize(block);
        }
        return;
    };

    inline py::array_t<table::galRow>
    process_image(
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        double variance,
        std::vector<geometry::block>& block_list,
        const std::optional<py::array_t<table::galRow>>& detection=std::nullopt,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt,
        const std::optional<py::array_t<int16_t>>& mask_array=std::nullopt
    ) {

        double variance_use;
        if (noise_array.has_value()) {
            variance_use = variance * 2.0;
        } else {
            variance_use = variance;
        }

        std::vector<table::galNumber> catalog;
        if (detection.has_value()) {
            catalog = table::array_to_objlist(
                *detection
            );
        } else {
            for (const geometry::block & block: block_list) {
                std::vector<table::galNumber> det = detect_block(
                    img_array,
                    psf_array,
                    variance_use,
                    block,
                    noise_array
                );
                catalog.reserve(catalog.size() + det.size());
                for (const table::galNumber& src : det) {
                    catalog.push_back(src);
                }
            }
        }

        for (geometry::block & block: block_list) {
            prepare_indices(
                catalog,
                block
            );
        }

        for (int run_id = 0; run_id < this->num_epochs_deblend; ++run_id) {
            std::vector<table::galNumber> catalog_model = catalog;
            for (table::galNumber & src : catalog_model) {
                src.model.F = src.model.F * src.wdet;
            }
            for (const geometry::block & block: block_list) {
                measure_block(
                    catalog,
                    catalog_model,
                    img_array,
                    psf_array,
                    variance_use,
                    block,
                    noise_array,
                    run_id
                );
            }
        }

        if (mask_array.has_value()) {
            mask::add_pixel_mask_column_catalog(
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
