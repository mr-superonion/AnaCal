#ifndef ANACAL_NGMIX_FITTING_H
#define ANACAL_NGMIX_FITTING_H

#include "../image.h"
#include "../math.h"
#include "../stdafx.h"
#include "../table.h"


namespace anacal {
namespace ngmix {

class GaussFit {
public:
    // stamp dimension
    double scale;
    double sigma_arcsec;
    int stamp_size, ss2;
    std::vector<double> grids_1d;

    GaussFit(
        double scale,
        double sigma_arcsec,
        int stamp_size=64
    ) : scale(scale), sigma_arcsec(sigma_arcsec), stamp_size(stamp_size),
        grids_1d(stamp_size, 0.0)
    {
        this->ss2 = stamp_size / 2;
        if (stamp_size % 2 != 0 ) {
            throw std::invalid_argument("nx or ny is not even number");
        }
        if (sigma_arcsec <= 0) {
            throw std::invalid_argument("sigma_arcsec must be positive");
        }
        for (int i = 0; i < this->stamp_size; ++i) {
            this->grids_1d[i] = (i - this->ss2) * this->scale;
            // range is [-ss2, ss2)
        }
    };

    inline math::lossNumber
    accumulate_loss(
        const std::vector<math::tnumber> & data,
        double variance,
        const NgmixGaussian & model,
        const geometry::block & block
    ) const {
        int i_stamp = static_cast<int>(
            std::round(model.x1.v / this->scale)
        );
        int j_stamp = static_cast<int>(
            std::round(model.x2.v / this->scale)
        );
        double x_stamp_shift = i_stamp * this->scale;
        double y_stamp_shift = j_stamp * this->scale;
        std::vector<double> xvs(stamp_size, 0.0);
        std::vector<double> yvs(stamp_size, 0.0);
        for (int i = 0; i < this->stamp_size; ++i){
            xvs[i] = this->grids_1d[i] + x_stamp_shift;
            yvs[i] = this->grids_1d[i] + y_stamp_shift;
        }
        int j_block_shift = j_stamp - this->ss2 - block.ymin;
        int i_block_shift = i_stamp - this->ss2 - block.xmin;

        modelKernel kernel = model.prepare_model(
            this->scale,
            this->sigma_arcsec
        );
        math::lossNumber loss;
        for (int j = 0; j < this->stamp_size; ++j) {
            int jb = j + j_block_shift;
            if (jb < 0 || jb >= block.ny) {
                continue;
            }
            int idjb = jb * block.nx;
            for (int i = 0; i < this->stamp_size; ++i) {
                int ib = i + i_block_shift;
                if (ib < 0 || ib >= block.nx) {
                    continue;
                }
                loss = loss + model.get_loss(
                    data[idjb + ib], variance, xvs[i], yvs[j], kernel
                );
            }
        }
        return loss;
    };

    inline std::vector<table::galNumber>
    process_block(
        const std::vector<table::galNumber>& catalog,
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        const modelPrior prior,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt,
        int num_epochs = 5,
        double variance = 1.0,
        std::optional<geometry::block> block=std::nullopt
    ) {

        int image_ny = img_array.shape(0);
        int image_nx = img_array.shape(1);
        geometry::block bb = block ? *block : geometry::get_block_list(
            image_nx, image_ny, image_nx, image_ny, 0, this->scale
        )[0];

        std::vector<math::tnumber> data = prepare_data_block(
            img_array,
            psf_array,
            this->sigma_arcsec,
            bb,
            noise_array
        );
        std::vector<table::galNumber> result;
        result.reserve(catalog.size());
        for (table::galNumber src : catalog) {
            for (int epoch = 0; epoch < num_epochs; ++epoch) {
                src.loss = this->accumulate_loss(
                    data, variance, src.model, bb
                );
                src.model.update_model_params(src.loss, prior);
            }
            result.push_back(src);
        }
        return result;
    };

};

} // end of ngmix
} // end of anacal

#endif // ANACAL_NGMIX_FITTING
