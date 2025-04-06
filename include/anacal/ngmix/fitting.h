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
    bool force_size, force_center;
    double fpfs_c0;
    double sigma2, sigma_m2, rfac, ffac, ffac2, ffac3;
    double sigma2_lim;

    GaussFit(
        double scale,
        double sigma_arcsec,
        int stamp_size=64,
        bool force_size=false,
        bool force_center=false,
        double fpfs_c0=1.0
    ) : scale(scale), sigma_arcsec(sigma_arcsec), stamp_size(stamp_size),
        ss2(stamp_size / 2), grids_1d(stamp_size, 0.0), force_size(force_size),
        force_center(force_center),
        fpfs_c0(fpfs_c0)
    {
        for (int i = 0; i < this->stamp_size; ++i) {
            this->grids_1d[i] = (i - this->ss2) * this->scale;
            // range is [-ss2, ss2)
        }
        this->sigma2 = sigma_arcsec * sigma_arcsec;
        this->sigma_m2 = 1.0 / this->sigma2;
        this->rfac = -0.5 * this->sigma_m2;
        this->ffac = rfac * (-0.318309886);
        this->ffac2 = this->ffac * 1.41421356 * this->sigma_m2;
        this->ffac3 = this->ffac * 2.0 * this->sigma_m2;
        this->sigma2_lim = sigma2 * 25 / scale / scale;
    };

    inline void
    measure_fpfs(
        const std::vector<math::qnumber> & data,
        table::galNumber & src,
        const geometry::block & block
    ) const {
        int i_stamp = static_cast<int>(
            std::round(src.model.x1.v / this->scale)
        );
        int j_stamp = static_cast<int>(
            std::round(src.model.x2.v / this->scale)
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

        math::qnumber m0, mxx, myy, mxy;
        for (int j = 0; j < this->stamp_size; ++j) {
            int jb = j + j_block_shift;
            if (jb < 0 || jb >= block.ny) {
                continue;
            }
            int idjb = jb * block.nx;
            int j2 = std::pow(j - this->ss2, 2);
            for (int i = 0; i < this->stamp_size; ++i) {
                int ib = i + i_block_shift;
                if (ib < 0 || ib >= block.nx) {
                    continue;
                }
                int i2 = std::pow(i - this->ss2, 2);
                if (i2 + j2 < this->sigma2_lim) {
                    std::array<math::qnumber, 4> mm = src.model.get_fpfs_moments(
                        data[idjb + ib], xvs[i], yvs[j], this->rfac
                    );
                    m0 = m0 + mm[0];
                    mxx = mxx + mm[1];
                    myy = myy + mm[2];
                    mxy = mxy + mm[3];
                }
            }
        }

        src.fpfs_m0 = m0 * this->ffac;
        src.fpfs_m2 = (mxx + myy - m0 * this->sigma2) * this->ffac3;
        {
            math::qnumber denom = (src.fpfs_m0 + this->fpfs_c0);
            src.fpfs_e1 = (mxx - myy) * this->ffac2 / denom;
            src.fpfs_e2 = 2.0 * mxy * this->ffac2 / denom;
        }
        // Orientation angle (in radians)
        src.model.t = 0.5 * math::atan2(src.fpfs_e2, src.fpfs_e1);
        return;
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
                math::lossNumber r2 = model.get_r2(xvs[i], yvs[j], kernel);
                if (r2.v.v < 30) {
                    loss = loss + model.get_loss(
                        data[idjb + ib], variance, r2, kernel
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
            if (!this->force_center) {
                // Do center refinement first
                src.model.force_size=true;
                src.model.force_center=false;
                for (int epoch = 0; epoch < num_epochs; ++epoch) {
                    src.loss = this->measure_loss(
                        data, variance, src.model, block
                    );
                    src.model.update_model_params(
                        src.loss, prior, epoch, variance
                    );
                }
            }
            // FPFS measurement
            this->measure_fpfs(
                data, src, block
            );
            if (!this->force_size) {
                src.model.force_size=false;
                src.model.force_center=true;
                for (int epoch = 0; epoch < num_epochs; ++epoch) {
                    src.loss = this->measure_loss(
                        data, variance, src.model, block
                    );
                    src.model.update_model_params(
                        src.loss, prior, epoch, variance
                    );
                }
            }
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
