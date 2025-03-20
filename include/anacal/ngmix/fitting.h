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
    bool force_size, force_shape, force_center;

    GaussFit(
        double scale,
        double sigma_arcsec,
        int stamp_size=64,
        bool force_size=false,
        bool force_shape=false,
        bool force_center=false
    ) : scale(scale), sigma_arcsec(sigma_arcsec), stamp_size(stamp_size),
        ss2(stamp_size / 2), grids_1d(stamp_size, 0.0), force_size(force_size),
        force_shape(force_shape), force_center(force_center)
    {
        for (int i = 0; i < this->stamp_size; ++i) {
            this->grids_1d[i] = (i - this->ss2) * this->scale;
            // range is [-ss2, ss2)
        }
    };

    inline std::array<math::qnumber, 4>
    measure_fpfs_2nd(
        const std::vector<math::qnumber> & data,
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

        std::array<math::qnumber, 4> result;
        for (int j = 0; j < this->stamp_size; ++j) {
            int jb = j + j_block_shift;
            if (jb < 0 || jb >= block.ny) {
                continue;
            }
            int idjb = jb * block.nx;
            int j2 = std::pow(j - this->ss2, 2.0);
            for (int i = 0; i < this->stamp_size; ++i) {
                int ib = i + i_block_shift;
                if (ib < 0 || ib >= block.nx) {
                    continue;
                }
                int i2 = std::pow(i - this->ss2, 2.0);
                if (i2 + j2 < this->ss2 * this->ss2) {
                    std::array<math::qnumber, 4> mm = model.get_fpfs_moments(
                        data[idjb + ib], xvs[i], yvs[j], this->sigma_arcsec
                    );
                    result[0] = result[0] + mm[0];
                    result[1] = result[1] + mm[1];
                    result[2] = result[2] + mm[2];
                    result[3] = result[3] + mm[3];
                }
            }
        }
        return result;
    };

    inline math::lossNumber
    measure_loss(
        const std::vector<math::qnumber> & data,
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
        /* std::cout<<model.force_shape<<std::endl; */

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
            int j2 = std::pow(j - this->ss2, 2.0);
            for (int i = 0; i < this->stamp_size; ++i) {
                int ib = i + i_block_shift;
                if (ib < 0 || ib >= block.nx) {
                    continue;
                }
                int i2 = std::pow(i - this->ss2, 2.0);
                if (i2 + j2 < this->ss2 * this->ss2) {
                    loss = loss + model.get_loss(
                        data[idjb + ib], variance, xvs[i], yvs[j], kernel
                    );
                }
            }
        }
        return loss;
    };

    inline std::vector<table::galNumber>
    process_block_impl(
        const std::vector<table::galNumber>& catalog,
        const std::vector<math::qnumber> & data,
        const modelPrior & prior,
        int num_epochs,
        double variance,
        geometry::block block
    ) {
        std::vector<table::galNumber> result;
        result.reserve(catalog.size());
        for (table::galNumber src : catalog) {
            src.model.force_size=this->force_size;
            src.model.force_shape=this->force_shape;
            src.model.force_center=this->force_center;
            // FPFS Shapes
            for (int epoch = 0; epoch < num_epochs; ++epoch) {
                src.loss = this->measure_loss(
                    data, variance, src.model, block
                );
                src.model.update_model_params(src.loss, prior, epoch, variance);
            }
            std::array<math::qnumber, 4> mm = this->measure_fpfs_2nd(
                data, src.model, block
            );
            src.fpfs_e1 = mm[1] / mm[0];
            src.fpfs_e2 = mm[2] / mm[0];
            src.fpfs_trace = mm[0] / mm[3];
            result.push_back(src);
        }
        return result;
    };

    inline std::vector<table::galNumber>
    process_block(
        const std::vector<table::galNumber>& catalog,
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        const modelPrior & prior,
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
        std::vector<math::qnumber> data = prepare_data_block(
            img_array,
            psf_array,
            this->sigma_arcsec,
            bb,
            noise_array
        );
        return process_block_impl(
            catalog,
            data,
            prior,
            num_epochs,
            variance,
            bb
        );
    };
};

} // end of ngmix
} // end of anacal

#endif // ANACAL_NGMIX_FITTING
