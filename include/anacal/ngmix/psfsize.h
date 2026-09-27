#ifndef ANACAL_NGMIX_PSFSIZE_H
#define ANACAL_NGMIX_PSFSIZE_H

#include "fitting.h"

namespace anacal {
namespace ngmix {

// Size of the round Gaussian re-smoothing kernel from the PSF, reproducing
// metadetect's default ``fitgauss`` target (metadetect/lsst/metacal_exposures
// _get_fitgauss_target_psf, without the separate 1 + 2 * step shear-step
// dilation that metacal applies on top):
//
//     sigma = sqrt(T d / 2)
//
// where (e1, e2, T) are the adaptive moments of the PSF image and d is the
// ellipticity dilation of _get_ellip_dilation.
//
// The adaptive moments are the best-fit elliptical Gaussian in the least-
// squares sense (Bernstein & Jarvis 2002), so they are measured here with
// the model fit: GaussFit's flux solve, loss and Gauss-Newton update, run
// on the RAW PSF image (no deconvolution, no re-smoothing).  The model is
// G(M + s^2 I) with a small kernel s, whose only role is to keep the
// covariance positive (the update's guard); the fitted Gaussian is the
// total covariance C = M + s^2 I, whatever s is.

// Kernel of the PSF fit relative to the pixel scale: small enough that
// M = C - s^2 I stays positive for any PSF sampled at >~ 1 pixel.
inline constexpr double psf_fit_kernel_rel = 0.25;

struct PsfGaussFit {
    double e1 = 0.0, e2 = 0.0, T = 0.0;
    double x1 = 0.0, x2 = 0.0, flux = 0.0;
    int n_epochs = 0;
    bool converged = false;
};

// metadetect _get_ellip_dilation: the eigenvalues of a covariance with
// ellipticity (e1, e2) and trace T are T / 2 (1 +- |e|), so
//     sqrt(lambda_max / (T / 2)) = sqrt(1 + |e|),
// doubled about 1 and capped at dilation_max (1.1 in metadetect).
inline double
ellip_dilation(double e1, double e2, double dilation_max = 1.1) {
    if (!(dilation_max >= 1.0)) {
        throw std::invalid_argument(
            "ellip_dilation: dilation_max must be >= 1"
        );
    }
    const double dil = 1.0 + 2.0 * (std::sqrt(1.0 + std::hypot(e1, e2)) - 1.0);
    return std::min(dil, dilation_max);
}

inline PsfGaussFit
fit_psf_gauss(
    const py::array_t<double>& psf_array,
    double scale,
    int max_epochs = 200,
    double tol = 1.0e-10
) {
    if (psf_array.ndim() != 2) {
        throw std::invalid_argument("fit_psf_gauss: psf_array must be 2D");
    }
    if (!(scale > 0.0)) {
        throw std::invalid_argument("fit_psf_gauss: scale must be > 0");
    }
    const int ny = static_cast<int>(psf_array.shape(0));
    const int nx = static_cast<int>(psf_array.shape(1));
    const geometry::cell cell = geometry::get_cell_list(
        nx, ny, nx, ny, 0, scale
    )[0];
    auto r = psf_array.unchecked<2>();
    // From here on only C++ containers and the unchecked accessor are
    // touched (the cell's py::array_t member above needed the GIL to
    // construct), so the fit runs GIL-free: xlens calls this from its
    // threaded cell loop.
    ScopedGilRelease release;
    // the cell grid is the array grid (get_cell_list pads nothing when
    // the cell is the whole image)
    std::vector<math::qnumber> data(static_cast<std::size_t>(nx) * ny);
    for (int j = 0; j < cell.ny; ++j) {
        const int jy = j + cell.ymin;
        for (int i = 0; i < cell.nx; ++i) {
            const int ix = i + cell.xmin;
            if (jy >= 0 && jy < ny && ix >= 0 && ix < nx) {
                data[static_cast<std::size_t>(j) * cell.nx + i] =
                    math::qnumber(r(jy, ix));
            }
        }
    }

    // Start: round, centred on pixel (nx // 2, ny // 2) (AnaCal's PSF
    // convention; also the centre of an odd DM kernel image), with the
    // size from a few Gaussian-weighted moment iterations (weight s_w^2
    // updated to the implied Gaussian variance, as adaptive moments do).
    const double x0 = (nx / 2) * scale;
    const double y0 = (ny / 2) * scale;
    double sw2 = 1.0;   // weight variance, arcsec^2
    for (int it = 0; it < 20; ++it) {
        double m0 = 0.0, mrr = 0.0;
        for (int j = 0; j < cell.ny; ++j) {
            const double dy = cell.yvs[j] - y0;
            for (int i = 0; i < cell.nx; ++i) {
                const double dx = cell.xvs[i] - x0;
                const double rr = dx * dx + dy * dy;
                const double w = std::exp(-0.5 * rr / sw2);
                const double f = w * data[static_cast<std::size_t>(j) * cell.nx + i].v;
                m0 += f;
                mrr += f * rr;
            }
        }
        if (!(m0 > 0.0)) {
            throw std::runtime_error(
                "fit_psf_gauss: PSF image has no positive flux near its centre"
            );
        }
        // Gaussian of variance c under a weight of variance sw2: the
        // weighted per-axis variance is v = c sw2 / (c + sw2)
        const double v = std::min(0.5 * mrr / m0, 0.9 * sw2);
        const double c = v * sw2 / (sw2 - v);
        if (std::abs(c - sw2) < 1.0e-6 * sw2) {
            sw2 = c;
            break;
        }
        sw2 = c;
    }

    const double s_k = psf_fit_kernel_rel * scale;
    const double s_k2 = s_k * s_k;
    // no priors, no misfit damping, no gate: the fixed point is the
    // unregularised least-squares Gaussian
    const GaussFit fit(
        scale, s_k, std::max(nx, ny), false, false,
        1.0, false,
        0.0, 0.5, 0.0, 0.0,
        0.05, 0.1, 0.0,
        0.0, 10.0
    );
    const modelPrior prior;
    table::galNumber src;
    src.model.force_size = false;
    src.model.force_center = false;
    src.model.x1 = math::qnumber(x0);
    src.model.x2 = math::qnumber(y0);
    const double m_init = std::max(sw2 - s_k2, 0.5 * s_k2);
    src.model.mxx = math::qnumber(m_init);
    src.model.myy = math::qnumber(m_init);
    src.model.mxy = math::qnumber(0.0);
    src.x1_det = x0;
    src.x2_det = y0;

    PsfGaussFit out;
    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        const double p0[5] = {
            src.model.mxx.v, src.model.myy.v, src.model.mxy.v,
            src.model.x1.v, src.model.x2.v
        };
        fit.solve_flux(data, 1.0, src, cell, prior);
        const modelKernelD kernel = src.model.prepare_modelD(scale, s_k);
        fit.measure_loss(data, 1.0, src, cell, kernel);
        // trust radii scaled to the PSF: 20% of its size per epoch for
        // the covariance, half a pixel for the centre
        const double tsize = 0.2 * (src.model.mxx.v + src.model.myy.v + 2.0 * s_k2);
        src.model.update_model_params(
            src.loss, prior, src.x1_det, src.x2_det,
            0.0, 0.0, 0.0,
            tsize, 0.5 * scale,
            0.0, 1.0, s_k2,
            math::qnumber(1.0)
        );
        out.n_epochs = epoch + 1;
        const double p1[5] = {
            src.model.mxx.v, src.model.myy.v, src.model.mxy.v,
            src.model.x1.v, src.model.x2.v
        };
        double dmax = 0.0;
        for (int k = 0; k < 5; ++k) {
            dmax = std::max(dmax, std::abs(p1[k] - p0[k]));
        }
        if (dmax < tol) {
            out.converged = true;
            break;
        }
    }
    fit.solve_flux(data, 1.0, src, cell, prior);

    const double cxx = src.model.mxx.v + s_k2;
    const double cyy = src.model.myy.v + s_k2;
    const double cxy = src.model.mxy.v;
    out.T = cxx + cyy;
    out.e1 = (cxx - cyy) / out.T;
    out.e2 = 2.0 * cxy / out.T;
    out.x1 = src.model.x1.v;
    out.x2 = src.model.x2.v;
    out.flux = src.model.F.v;
    return out;
}

// sigma [arcsec] of metadetect's fitgauss target for this PSF image
inline double
fitgauss_sigma_arcsec(
    const py::array_t<double>& psf_array,
    double scale,
    double dilation_max = 1.1
) {
    const PsfGaussFit r = fit_psf_gauss(psf_array, scale);
    if (!r.converged || !(r.T > 0.0)) {
        throw std::runtime_error(
            "fitgauss_sigma_arcsec: the Gaussian fit to the PSF failed"
        );
    }
    const double d = ellip_dilation(r.e1, r.e2, dilation_max);
    return std::sqrt(r.T * d / 2.0);
}

} // end of ngmix
} // end of anacal

#endif // ANACAL_NGMIX_PSFSIZE_H
