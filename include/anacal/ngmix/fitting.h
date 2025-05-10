#ifndef ANACAL_NGMIX_FITTING_H
#define ANACAL_NGMIX_FITTING_H

#include "../image.h"
#include "../math.h"
#include "../stdafx.h"
#include "../table.h"


namespace anacal {
namespace ngmix {


// model fitting upper scale 3.5 arcsec
// deblending upper scale 7 arcsec

class GaussFit {
public:
    // stamp dimension
    double scale;
    double sigma_arcsec;
    int stamp_size, ss2;
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
        ss2(stamp_size / 2), force_size(force_size),
        force_center(force_center),
        fpfs_c0(fpfs_c0)
    {
        this->sigma2 = sigma_arcsec * sigma_arcsec;
        this->sigma_m2 = 1.0 / this->sigma2;
        this->rfac = -0.5 * this->sigma_m2;
        this->ffac = rfac * (-0.318309886);
        this->ffac2 = this->ffac * 1.41421356 * this->sigma_m2;
        this->ffac3 = this->ffac * 2.0 * this->sigma_m2;
        this->sigma2_lim = sigma2 * 20;
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
    add_model_to_block(
        std::vector<math::qnumber> & data_model,
        const table::galNumber & src,
        const geometry::block & block,
        const modelKernelB & kernel
    ) const {
        const ngmix::NgmixGaussian & model = src.model;
        const StampBounds bb = model.get_stamp_bounds(block, 7 / block.scale);
        for (int j = bb.j_min; (j < bb.j_max); ++j) {
            if (!block.ymsk[j]) continue;
            int jj = j * block.nx;
            for (int i = bb.i_min; (i < bb.i_max); ++i) {
                if (!block.xmsk[i]) continue;
                if (bb.has_point(i, j)) {
                    math::qnumber r2 = model.get_r2(
                        block.xvs[i], block.yvs[j], kernel
                    );
                    data_model[jj + i] = (
                        data_model[jj + i]
                        + src.model.get_model_from_r2(r2, kernel) * src.wdet
                    );
                }
            }
        }
        return;
    };

    inline void
    measure_flux(
        const std::vector<math::qnumber> & data,
        table::galNumber & src,
        const geometry::block & block,
        const modelKernelD & kernel
    ) const {
        math::qnumber m0, norm;
        ngmix::NgmixGaussian & model = src.model;
        const StampBounds bb = model.get_stamp_bounds(block, 3.5 / block.scale);
        for (int j = bb.j_min; (j < bb.j_max); ++j) {
            if (!block.ymsk[j]) continue;
            int jj = j * block.nx;
            for (int i = bb.i_min; (i < bb.i_max); ++i) {
                if (!block.xmsk[i]) continue;
                if (bb.has_point(i, j)) {
                    math::lossNumber r2 = model.get_r2(
                        block.xvs[i], block.yvs[j], kernel
                    );
                    math::qnumber w = src.model.get_func_from_r2(
                        r2,
                        kernel
                    );
                    norm = norm + w * w;
                    m0 = m0 + w * data[jj + i];
                }
            }
        }
        src.model.F = m0 / norm;
        return;
    };

    inline void
    measure_loss(
        const std::vector<math::qnumber> & data,
        double variance,
        table::galNumber & src,
        const geometry::block & block,
        const modelKernelD & kernel
    ) const {
        src.loss.reset();
        ngmix::NgmixGaussian & model = src.model;
        const StampBounds bb = model.get_stamp_bounds(block, 3.5 / block.scale);

        for (int j = bb.j_min; (j < bb.j_max); ++j) {
            if (!block.ymsk[j]) continue;
            int jj = j * block.nx;
            for (int i = bb.i_min; (i < bb.i_max); ++i) {
                if (!block.xmsk[i]) continue;
                if (bb.has_point(i, j)) {
                    math::lossNumber r2 = model.get_r2(
                        block.xvs[i], block.yvs[j], kernel
                    );
                    src.loss = src.loss + model.get_loss(
                        data[jj + i], variance, r2, kernel
                    );
                }
            }
        }
        return;
    };

    inline void
    measure_loss2(
        const std::vector<math::qnumber> & data,
        const std::vector<math::qnumber> & data_m,
        double variance,
        table::galNumber & src,
        const modelKernelD & kernel,
        const NgmixGaussian & mmod,
        const modelKernelB & mkernel,
        const geometry::block & block
    ) const {
        src.loss.reset();
        NgmixGaussian & model = src.model;
        const StampBounds bb = model.get_stamp_bounds(block, 3.5 / block.scale);

        for (int j = bb.j_min; (j < bb.j_max); ++j) {
            if (!block.ymsk[j]) continue;
            int jj = j * block.nx;
            for (int i = bb.i_min; (i < bb.i_max); ++i) {
                if (!block.xmsk[i]) continue;
                if (bb.has_point(i, j)) {
                    math::qnumber p = data_m[jj+i];
                    const math::qnumber mr2 = mmod.get_r2(
                        block.xvs[i], block.yvs[j], mkernel
                    );
                    p = p - mmod.get_model_from_r2(mr2, mkernel) * src.wdet;
                    const math::lossNumber r2 = model.get_r2(
                        block.xvs[i], block.yvs[j], kernel
                    );
                    src.loss = src.loss + model.get_loss_with_p(
                        data[jj + i], variance, r2, kernel, p
                    );
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

        ngmix::NgmixGaussian & model = src.model;
        int r = static_cast<int>(this->sigma_arcsec * 8 / block.scale);
        int i_min = std::max(
            static_cast<int>(
                std::round(model.x1.v / this->scale)
            ) - block.xmin - r,
            0
        );
        int i_max = std::min(i_min + 2 * r + 1, block.nx);
        int j_min = std::max(
            static_cast<int>(
                std::round(model.x2.v / this->scale)
            ) - block.ymin - r,
            0
        );
        int j_max = std::min(j_min + 2 * r + 1, block.ny);

        math::qnumber m0, mxx, myy, mxy;
        for (int j = j_min; j < j_max; ++j) {
            int jj = j * block.nx;
            double ys = block.yvs[j] - model.x2.v;
            double y2 = ys * ys;
            for (int i = i_min; i < i_max; ++i) {
                double xs = block.xvs[i] - model.x1.v;
                double x2 = xs * xs;
                if ((x2 + y2) < this->sigma2_lim) {
                    std::array<math::qnumber, 4> mm = src.model.get_fpfs_moments(
                        data[jj + i],
                        block.xvs[i],
                        block.yvs[j],
                        this->rfac
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

    inline void
    initialize_fitting(
        const std::vector<math::qnumber> & data,
        NgmixGaussian & model,
        const geometry::block & block
    ) const {
        math::qnumber m0, mxx, myy, mxy, norm;
        double a_ini = 0.15;
        model.a1 = a_ini;
        model.a2 = a_ini;
        double dd = 1.0 / (this->sigma2 + a_ini * a_ini);

        const StampBounds bb = model.get_stamp_bounds(block, 3.5 / block.scale);

        for (int j = bb.j_min; (j < bb.j_max); ++j) {
            if (!block.ymsk[j]) continue;
            int jj = j * block.nx;
            math::qnumber ys = block.yvs[j] - model.x2.v;
            math::qnumber y2 = math::pow(ys, 2);
            for (int i = bb.i_min; (i < bb.i_max); ++i) {
                if (!block.xmsk[i]) continue;
                math::qnumber xs = block.xvs[i] - model.x1.v;
                math::qnumber x2 = math::pow(xs, 2);
                math::qnumber xy = xs * ys;
                math::qnumber r2 = (x2 + y2) * dd;
                if (bb.has_point(i, j)) {
                    math::qnumber w = math::exp(-0.5 * r2);
                    math::qnumber f = (
                        w * data[jj + i]
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
        /* model.F = m0 * (4.0 * M_PI * this->sigma2) / norm / block.scale / block.scale; */
        return;
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
        const std::optional<py::array_t<double>>& noise_array=std::nullopt,
        const std::optional<std::vector<std::size_t>>& indices=std::nullopt,
        std::optional<int> run_id=std::nullopt
    ) {
        int irun;
        if (run_id.has_value()) {
            irun = *run_id;
        } else {
            irun = 0;
        }

        std::vector<std::size_t> inds;
        if (indices.has_value()) {
            inds = *indices;
        } else {
            std::size_t ns = catalog.size();
            inds.reserve(ns);
            for (std::size_t i = 0; i < ns; ++i) {
                inds.push_back(i);
            }
        }

        std::vector<math::qnumber> data = prepare_data_block(
            img_array,
            psf_array,
            this->sigma_arcsec,
            block,
            noise_array
        );

        std::vector<NgmixGaussian> catalog_model;
        std::vector<modelKernelB> kernels_model;
        std::size_t ng = inds.size();
        std::vector<math::qnumber> data_model;
        std::size_t n_pix = data.size();

        if (irun == 0) {
            // initialize the sources
            for (const std::size_t ind : inds) {
                table::galNumber & src = catalog[ind];
                src.model.force_size=this->force_size;
                src.model.force_center=this->force_center;
                if (
                    (!this->force_size) &&
                    (!src.initialized) &&
                    (src.block_id == block.index)
                ) {
                    initialize_fitting(data, src.model, block);
                    src.initialized = true;
                }
            }
        } else {
            catalog_model.reserve(ng);
            kernels_model.reserve(ng);
            data_model.resize(n_pix);
            for (std::size_t i=0; i<ng; ++i) {
                const table::galNumber mrc = catalog[inds[i]];
                catalog_model.push_back(mrc.model);
                const modelKernelB kernel = mrc.model.prepare_modelB(
                    this->scale,
                    this->sigma_arcsec
                );
                kernels_model.push_back(kernel);
                add_model_to_block(data_model, mrc, block, kernel);
            }
        }

        double variance_meas = get_smoothed_variance(
            block.scale,
            this->sigma_arcsec,
            psf_array,
            variance
        );
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            /* std::cout<<"epoch: "<<epoch<<std::endl; */
            for (std::size_t i=0; i<ng; ++i) {
                table::galNumber & src = catalog[inds[i]];
                if (src.block_id == block.index) {
                    const modelKernelD kernel = src.model.prepare_modelD(
                        this->scale,
                        this->sigma_arcsec
                    );
                    if (irun == 0) {
                        this->measure_loss(
                            data, variance_meas, src, block, kernel
                        );
                    } else {
                        const NgmixGaussian & mmod = catalog_model[i];
                        const modelKernelB & mkernel = kernels_model[i];
                        this->measure_loss2(
                            data, data_model,
                            variance_meas,
                            src, kernel,
                            mmod, mkernel,
                            block
                        );
                    }
                    src.model.update_model_params(
                        src.loss, prior, variance_meas
                    );
                }
            }
        }

        if (irun==2) {
            // finally get FPFS measurement
            double std_fpfs = std::sqrt(
                get_smoothed_variance(
                    block.scale,
                    this->sigma_arcsec * 1.414,
                    psf_array,
                    variance
                )
            );

            for (std::size_t i=0; i<ng; ++i) {
                table::galNumber & src = catalog[inds[i]];
                if (src.block_id == block.index) {
                    this->measure_fpfs(
                        data, src, block
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
            }
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
