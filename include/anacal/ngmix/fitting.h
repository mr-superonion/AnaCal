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
    double ap2_r, ap2_r2;
    double r2_lim_stamp;

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
        this->sigma2_lim = sigma2 * 20 / scale / scale;
        this->ap2_r = 2.0 / scale;
        this->ap2_r2 = std::pow(ap2_r, 2.0);
        this->r2_lim_stamp = std::pow((this->ss2-1) * scale, 2.0);
    };

    inline void
    measure_aperture_flux(
        const std::vector<math::qnumber> & data,
        table::galNumber & src,
        const geometry::block & block
    ) const {
        // stamp center index
        int i = static_cast<int>(
            std::round(src.model.x1.v / this->scale)
        ) - block.xmin;
        int j = static_cast<int>(
            std::round(src.model.x2.v / this->scale)
        ) - block.ymin;

        math::qnumber fluxap2;
        for (int dj = -ap2_r; dj <= ap2_r; dj++) {
            int dj2 = dj * dj;
            for (int di = -ap2_r; di <= ap2_r; di++) {
                int dr2 = di * di + dj2;
                if (dr2 < this->ap2_r2) {
                    int _i = (j + dj) * block.nx + (i + di);
                    src.fluxap2 = src.fluxap2 + data[_i];
                }
            }
        }
        return;
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
        int i_block_shift = i_stamp - this->ss2 - block.xmin;
        int j_block_shift = j_stamp - this->ss2 - block.ymin;

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
        return;
    };

    /* inline void */
    /* measure_admom( */
    /*     const std::vector<math::qnumber> & data, */
    /*     NgmixGaussian & model, */
    /*     const geometry::block & block */
    /* ) const { */
    /*     int i_stamp = static_cast<int>( */
    /*         std::round(model.x1.v / this->scale) */
    /*     ); */

    /*     int j_stamp = static_cast<int>( */
    /*         std::round(model.x2.v / this->scale) */
    /*     ); */
    /*     double x_stamp_shift = i_stamp * this->scale; */
    /*     double y_stamp_shift = j_stamp * this->scale; */
    /*     std::vector<double> xvs(stamp_size, 0.0); */
    /*     std::vector<double> yvs(stamp_size, 0.0); */
    /*     for (int i = 0; i < this->stamp_size; ++i){ */
    /*         xvs[i] = this->grids_1d[i] + x_stamp_shift; */
    /*         yvs[i] = this->grids_1d[i] + y_stamp_shift; */
    /*     } */
    /*     int j_block_shift = j_stamp - this->ss2 - block.ymin; */
    /*     int i_block_shift = i_stamp - this->ss2 - block.xmin; */

    /*     int n_iter_admom = 2; */
    /*     for (int it=0; it<n_iter_admom; ++it){ */
    /*         model.update_model_admom_inv(this->sigma_arcsec); */
    /*         math::qnumber m0, mx, my, mxx, myy, mxy, norm; */
    /*         for (int j = 0; j < this->stamp_size; ++j) { */
    /*             int jb = j + j_block_shift; */
    /*             if (jb < 0 || jb >= block.ny) { */
    /*                 continue; */
    /*             } */
    /*             int idjb = jb * block.nx; */
    /*             for (int i = 0; i < this->stamp_size; ++i) { */
    /*                 int ib = i + i_block_shift; */
    /*                 if (ib < 0 || ib >= block.nx) { */
    /*                     continue; */
    /*                 } */
    /*                 math::qnumber xs = xvs[i] - model.x1; */
    /*                 math::qnumber ys = yvs[j] - model.x2; */
    /*                 math::qnumber x2 = math::pow(xs, 2); */
    /*                 math::qnumber y2 = math::pow(ys, 2); */
    /*                 math::qnumber xy = xs * ys; */
    /*                 math::qnumber r2 = ( */
    /*                     x2 * model.dxx + 2.0 * xy * model.dxy + y2 * model.dyy */
    /*                 ); */
    /*                 if ((r2.v < 20) & ((x2.v + y2.v) < this->r2_lim_stamp )) { */
    /*                     math::qnumber w = math::exp6(-0.5 * r2); */
    /*                     math::qnumber f = ( */
    /*                         w * data[idjb + ib] */
    /*                     ); */
    /*                     norm = norm + w * w; */
    /*                     m0 = m0 + f; */
    /*                     mxx = mxx + f * x2; */
    /*                     myy = myy + f * y2; */
    /*                     mxy = mxy + f * xy; */
    /*                     mx = mx + f * xs; */
    /*                     my = mx + f * ys; */
    /*                 } */
    /*             } */
    /*         } */
    /*         math::qnumber rr = 1.0 / m0; */
    /*         model.x1 = model.x1 + mx * rr / n_iter_admom; */
    /*         model.x2 = model.x2 + my * rr / n_iter_admom; */
    /*         model.update_model_admom( */
    /*             this->sigma_arcsec, */
    /*             mxx * rr, myy * rr, mxy * rr */
    /*         ); */
    /*         model.F = m0 * (2.0 * M_PI / math::sqrt(model.idet)) / norm / block.scale / block.scale; */
    /*         model.t = 0.5 * math::atan2(2.0 * mxy, mxx - myy); */
    /*         math::qnumber delta = math::sqrt( */
    /*             math::pow(mxx - myy, 2) */
    /*             + 4.0 * math::pow(mxy, 2) */
    /*         ); */
    /*         if (mxx.v + myy.v - delta.v > 1e-10) { */
    /*             model.a1 = math::sqrt(0.5 * (mxx + myy + delta)); */
    /*             model.a2 = math::sqrt(0.5 * (mxx + myy - delta)); */
    /*         } else { */
    /*             model.a1 = math::qnumber(0.0); */
    /*             model.a2 = math::qnumber(0.0); */
    /*         } */
    /*     } */

    /*     return; */
    /* }; */

    inline void
    initialize_fitting(
        const std::vector<math::qnumber> & data,
        NgmixGaussian & model,
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

        math::qnumber m0, mxx, myy, mxy, norm;
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
                math::qnumber xs = xvs[i] - model.x1;
                math::qnumber ys = yvs[j] - model.x2;
                math::qnumber x2 = math::pow(xs, 2);
                math::qnumber y2 = math::pow(ys, 2);
                math::qnumber xy = xs * ys;
                double dd = 0.5 / this->sigma2;
                math::qnumber r2 = (x2 + y2) * dd;
                if (r2.v < 20) {
                    math::qnumber w = math::exp6(-0.5 * r2);
                    math::qnumber f = (
                        w * data[idjb + ib]
                    );
                    norm = norm + w * w;
                    m0 = m0 + f;
                    mxx = mxx + f * x2;
                    myy = myy + f * y2;
                    mxy = mxy + f * xy;
                }
            }
        }
        model.t = 0.5 * math::atan2(2.0 * mxy, mxx - myy);
        model.F = m0 * (4.0 * M_PI * this->sigma2) / norm / block.scale / block.scale;
        model.a1 = sigma_arcsec;
        model.a2 = sigma_arcsec;
        return;
    };

    inline math::lossNumber
    measure_loss(
        const std::vector<math::qnumber> & data,
        double variance,
        const NgmixGaussian & model,
        const geometry::block & block,
        const modelKernelD & kernel
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

                double xs = xvs[i] - model.x1.v;
                double ys = yvs[j] - model.x2.v;
                if (r2.v.v < 20 & xs * xs + ys * ys < this->r2_lim_stamp) {
                    loss = loss + model.get_loss(
                        data[idjb + ib], variance, r2, kernel
                    );
                }
            }
        }
        return loss;
    };

    inline void
    process_block_impl(
        std::vector<table::galNumber>& catalog,
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        const modelPrior & prior,
        int num_epochs,
        double variance,
        geometry::block block,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt
    ) {

        std::vector<math::qnumber> data = prepare_data_block(
            img_array,
            psf_array,
            this->sigma_arcsec,
            block,
            noise_array
        );
        double variance_meas = get_smoothed_variance(
            block.scale,
            this->sigma_arcsec,
            psf_array,
            variance
        );

        std::vector<modelKernelD> kernels;
        size_t ns = catalog.size();
        kernels.reserve(ns);
        for (table::galNumber & src : catalog) {
            src.model.force_size=this->force_size;
            src.model.force_center=this->force_center;
            if (!this->force_size) {
                initialize_fitting(data, src.model, block);
            }
            kernels.emplace_back(
                src.model.prepare_modelD(
                    this->scale,
                    this->sigma_arcsec
                )
            );
        }

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            for (size_t ind = 0; ind < ns; ++ind) {
                table::galNumber & src = catalog[ind];
                modelKernelD & kernel = kernels[ind];
                src.loss = this->measure_loss(
                    data, variance_meas, src.model, block, kernel
                );
                src.model.update_model_params(
                    src.loss, prior, variance_meas
                );
                kernel = src.model.prepare_modelD(
                    this->scale,
                    this->sigma_arcsec
                );
            }
        }

        for (table::galNumber & src : catalog) {
            this->measure_fpfs(
                data, src, block
            );
            double std_fpfs = std::pow(
                get_smoothed_variance(
                    block.scale,
                    this->sigma_arcsec * 1.414,
                    psf_array,
                    variance
                ),
                0.5
            );
            src.wsel = src.wdet * math::ssfunc1(
                src.fpfs_m0,
                5.0 * std_fpfs,
                std_fpfs
            );
            src.wsel = src.wsel * math::ssfunc1(
                src.fpfs_m2 - 0.1 * src.fpfs_m0,
                std_fpfs,
                std_fpfs
            );
        }
        return;
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
        std::vector<table::galNumber> result = catalog;
        process_block_impl(
            result,
            img_array,
            psf_array,
            prior,
            num_epochs,
            variance,
            bb,
            noise_array
        );
        return result;
    };
};

} // end of ngmix
} // end of anacal

#endif // ANACAL_NGMIX_FITTING
