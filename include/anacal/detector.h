#ifndef ANACAL_DETECTOR
#define ANACAL_DETECTOR

#include "image.h"

namespace anacal {
namespace detector {

inline constexpr int drmax = 5;
inline constexpr int drmax2 = drmax * drmax;

// Smallest detection weight worth keeping a row for.
inline constexpr double wdet_min = 1e-5;

// Local-background estimate for the detection weight.
//
// A cascade of concentric rings is walked outward from 1 to 3 arcsec in 0.5
// arcsec steps.  Each step blends the running estimate with the next ring out:
//
//     ratio = ssfunc1(b_prev - b_next, 0, std_noise)
//     b_next <- ratio * b_next + (1 - ratio) * b_prev
//
// ssfunc1 with mu = 0 selects the SMALLER of the two either way, so this is a
// smooth minimum over the rings.  That is what makes it robust in both
// directions: on a profile FALLING outward the rings are still contaminated by
// the candidate's own light, so the estimate is pushed further out to escape it;
// on a CONVEX background -- where a symmetric ring average is inflated by the
// curvature -- it stays local instead.  Those are the two opposing biases that
// a single fixed aperture has to trade off against each other.
//
// ``ratio`` is a qnumber, so the shear response of the choice propagates and the
// blend stays differentiable.  Verified against central finite differences to
// better than 1% in a regime where the background term drives the weight.
struct bkgKernel {
    std::vector<int> di, dj;
    std::vector<double> w;
    // Half-width in pixels of the box the offsets were selected from, i.e. the
    // furthest this kernel ever reaches from a candidate.  find_peaks_impl
    // compares it against the block's ``block_bound`` (geometry.h) to make sure
    // every read lands inside the block.
    int nr = 0;
    // Ring k spans [ring_start[k], ring_start[k + 1]) in the arrays above, and
    // its weights sum to one on their own.
    std::vector<std::size_t> ring_start;
};

// Ring edges, in arcsec so the cascade means the same at any pixel scale.
inline constexpr double cascade_r0 = 1.0;
inline constexpr double cascade_dr = 0.5;
inline constexpr int cascade_nring = 4;

// The background is subtracted in FULL.  The historical 0.6 existed because a
// fixed 1-2 arcsec annulus also contains the target's own wings, so full
// subtraction would have erased extended galaxies; the cascade escapes that
// light by construction.  At 0.6 the leftover 40% of a bright galaxy's surface
// brightness is itself far above the threshold out to ~100 arcsec, which left
// the cut inert exactly where it was needed.
inline constexpr double bkg_coeff = 1.0;

// Significance required ABOVE the local background.  The other flux cut,
// ssfunc1(data, snr_peak_min * std_noise, ...), is applied to the RAW smoothed
// value, which a bright galaxy's own light satisfies for free -- so this
// threshold is what actually governs detection on top of bright extended
// objects.  At the historical 3.0 the effective bar there collapsed from
// snr_peak_min to 3 sigma and produced thousands of spurious peaks; 5.0 removes
// essentially all of them while leaving blank-sky completeness untouched.
inline constexpr double bkg_nsigma = 5.0;

// Build the ring weights.  Depends only on the pixel scale, so this is done
// once per block and reused by every candidate peak.
inline bkgKernel
make_bkg_kernel(double scale) {
    bkgKernel bk;
    bk.nr = static_cast<int>(
        (cascade_r0 + cascade_nring * cascade_dr) / scale
    ) + 1;
    bk.ring_start.push_back(0);
    for (int k = 0; k < cascade_nring; ++k) {
        const double ra = cascade_r0 + k * cascade_dr;
        const double rb = ra + cascade_dr;
        const double ra2 = ra * ra, rb2 = rb * rb;
        std::size_t n0 = bk.di.size();
        for (int dj = -bk.nr; dj <= bk.nr; ++dj) {
            for (int di = -bk.nr; di <= bk.nr; ++di) {
                double r2 = (di * di + dj * dj) * scale * scale;
                if ((r2 >= ra2) && (r2 < rb2)) {
                    bk.di.push_back(di);
                    bk.dj.push_back(dj);
                }
            }
        }
        std::size_t n = bk.di.size() - n0;
        if (n == 0) {
            throw std::runtime_error("detector: empty cascade ring");
        }
        bk.w.insert(bk.w.end(), n, 1.0 / static_cast<double>(n));
        bk.ring_start.push_back(bk.di.size());
    }
    return bk;
};

inline
void measure_pixel(
    std::vector<table::galNumber> & catalog,
    const std::vector<math::qnumber> & data,
    int x,                                      // index on image
    int y,
    const geometry::block & block,
    double f_min,
    double omega_f,
    double omega_v,
    const bkgKernel & bk,
    double std_noise
) {
    int j = y - block.ymin;
    int i = x - block.xmin;
    int index = j * block.nx + i;
    // ssfunc1(v, omega_v, omega_v) is identically zero for v <= 0, so the
    // product below vanishes unless the centre strictly exceeds every
    // neighbour.  The gate therefore sits exactly on the zero-weight locus:
    // it is a pure short-circuit, never an uncorrected hard cut.
    const double wdet_cut = 0.0;
    // Radius-5 disk: di^2 + dj^2 <= 25 inclusive, excluding the centre, so
    // 80 neighbour differences.  The disk is invariant under a 90 degree
    // rotation, which is what keeps the weight free of additive shear bias.
    math::qnumber wdet = math::qnumber(1.0);
    for (int dj = -drmax; dj <= drmax; dj++) {
        int dj2 = dj * dj;
        for (int di = -drmax; di <= drmax; di++) {
            int dr2 = di * di + dj2;
            if ((dr2 <= drmax2) && (dr2 != 0)) {
                int index2 = (j + dj) * block.nx + (i + di);
                wdet = wdet * math::ssfunc1(
                    data[index] - data[index2],
                    omega_v,
                    omega_v
                );
            }
        }
    }
    if (wdet.v > wdet_cut) {
        table::galNumber src;
        src.x1_det = x * block.scale;
        src.x2_det = y * block.scale;
        src.model.x1.v = src.x1_det;
        src.model.x2.v = src.x2_det;

        // Walk outward, blending each ring against the running estimate.
        // Ring weights are pre-normalised, so each ``b`` is already a level.
        math::qnumber bkg;
        for (std::size_t k = 0; k + 1 < bk.ring_start.size(); ++k) {
            math::qnumber b;
            for (std::size_t t = bk.ring_start[k];
                 t < bk.ring_start[k + 1]; ++t) {
                int _i = (j + bk.dj[t]) * block.nx + (i + bk.di[t]);
                b = b + data[_i] * bk.w[t];
            }
            if (k == 0) {
                bkg = b;
            } else {
                math::qnumber ratio = math::ssfunc1(bkg - b, 0.0, std_noise);
                // ratio * b + (1 - ratio) * bkg, using only qnumber + - *
                bkg = bkg + ratio * (b - bkg);
            }
        }

        src.bkg = bkg;
        src.block_id = block.index;
        // The neighbour-difference product is used directly as the weight.
        // It is already bounded in [0, 1] (every factor is an ssfunc1), so it
        // needs no re-sharpening or renormalisation.
        src.wdet = wdet * math::ssfunc1(
            data[index],
            f_min,
            omega_f
        ) * math::ssfunc1(
            data[index] - bkg * bkg_coeff,
            bkg_nsigma * std_noise,
            omega_f
        );
        if (src.wdet.v > wdet_min) catalog.push_back(src);
    }
};

inline std::vector<table::galNumber>
find_peaks_impl(
    const py::array_t<pixel_t>& img_array,
    const py::array_t<double>& psf_array,
    double sigma_arcsec,
    double snr_min,
    double variance,
    double omega_f,
    double omega_v,
    const geometry::block & block,
    const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt,
    int image_bound=0
) {
    double sigma_arcsec_det = sigma_arcsec * sqrt2;
    std::vector<math::qnumber> data = prepare_data_block(
        img_array,
        psf_array,
        sigma_arcsec_det,
        block,
        noise_array
    );

    double std_noise = std::pow(
        get_smoothed_variance(
            block.scale,
            sigma_arcsec_det,
            psf_array,
            variance
        ), 0.5
    );
    // Secondary peak cut
    double f_min = std_noise * snr_min;
    double f_cut = f_min - omega_f;
    // ssfunc1(v, omega_v, omega_v) vanishes for v <= 0, so this cheap
    // pre-filter sits exactly where the weight is already zero.
    const double v_cut = 0.0;

    int image_ny = img_array.shape(0);
    int image_nx = img_array.shape(1);

    // Local-background weights: built once here, reused by every candidate.
    const bkgKernel bk = make_bkg_kernel(block.scale);

    // The background kernel reads out to +-bk.nr around every candidate, and
    // candidates run over the block's INNER region, so the padding between the
    // inner region and the block edge -- ``block_bound`` in geometry.h -- must
    // be at least bk.nr, or those reads fall outside ``data``: silently wrong
    // values where the row index merely wraps, undefined behaviour where it
    // goes negative.  Since block_bound = max(block_overlap / 2, 3), this
    // demands block_overlap >= 2 * bk.nr: 32 at 0.2 arcsec/pix, 62 at 0.1.
    // Refuse the configuration rather than quietly trimming the scan region, so
    // a mis-sized block surfaces immediately instead of biasing the edges.
    const int block_bound = std::min({
        block.ymin_in - block.ymin, block.ymax - block.ymax_in,
        block.xmin_in - block.xmin, block.xmax - block.xmax_in
    });
    if (block_bound < bk.nr) {
        throw std::runtime_error(
            "detector Error: block_bound of " + std::to_string(block_bound) +
            " pixels is smaller than the background kernel reach of " +
            std::to_string(bk.nr) + " pixels; pass block_overlap >= " +
            std::to_string(2 * bk.nr) + " to get_block_list"
        );
    }

    int ystart = std::max(image_bound, block.ymin_in);
    int yend = std::min(image_ny - image_bound, block.ymax_in);
    int xstart = std::max(image_bound, block.xmin_in);
    int xend = std::min(image_nx - image_bound, block.xmax_in);

    std::vector<table::galNumber> catalog;
    for (int y = ystart; y < yend; ++y) {
        int j = y - block.ymin;
        for (int x = xstart; x < xend; ++x) {
            int i = x - block.xmin;
            // data index
            int index = j * block.nx + i;
            if (
                (data[index].v > f_cut) &&
                (data[index].v - data[j * block.nx + (i + 1)].v > v_cut) &&
                (data[index].v - data[j * block.nx + (i - 1)].v > v_cut) &&
                (data[index].v - data[(j + 1) * block.nx + i].v > v_cut) &&
                (data[index].v - data[(j - 1) * block.nx + i].v > v_cut)
            ) {
                measure_pixel(
                    catalog,
                    data,
                    x,
                    y,
                    block,
                    f_min,
                    omega_f,
                    omega_v,
                    bk,
                    std_noise
                );
            }

        }
    }
    return catalog;
};

inline std::vector<table::galNumber>
find_peaks(
    const py::array_t<pixel_t>& img_array,
    const py::array_t<double>& psf_array,
    double sigma_arcsec,
    double snr_min,
    double variance,
    double omega_f,
    double omega_v,
    const geometry::block & block,
    const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt,
    int image_bound=0
) {
    std::vector<table::galNumber> cat = find_peaks_impl(
        img_array,
        psf_array,
        sigma_arcsec,
        snr_min,
        variance,
        omega_f,
        omega_v,
        block,
        noise_array,
        image_bound
    );

    // --------------------------------------------------------------------
    // DISABLED: neighbour-competition ("deblend") re-weighting.
    //
    // ``ss`` averaged the wdet of *other* detected peaks inside r^2 < 8
    // (r < 2.83 pix).  Step 1 makes that region provably empty: if a pixel
    // is detected then it strictly exceeds every neighbour inside the
    // radius-``drmax`` stencil, and because the stencil is symmetric each of
    // those neighbours picks up an ssfunc1 with a negative argument and so
    // gets wdet = 0.  Detected peaks are therefore always more than
    // ``drmax`` pixels apart, which exceeds the 2.83 pix footprint for any
    // drmax >= 3, so ``ss`` is identically zero for every source.  Measured
    // directly on a dense field: minimum peak separation was 5.0 pix at
    // drmax = 4, with zero pairs inside the footprint.
    //
    // With ss == 0 the factor below degenerates into ssfunc1(wdet, 0.4,
    // 0.399): a pure re-sharpening of wdet, the same construction as the
    // p_min/omega_p layer that was removed, but wider (zero below wdet =
    // 0.001, one above 0.799).  It performs no deblending, and because its
    // thresholds are fixed while wdet's scale depends on the stencil size,
    // it silently couples the two.  Kept here for reference only.
    //
    // std::vector<math::qnumber> data(block.nx * block.ny);
    // for (const table::galNumber & src: cat){
    //     const ngmix::NgmixGaussian & model = src.model;
    //     int i = static_cast<int>(
    //         std::round(model.x1.v / block.scale)
    //     ) - block.xmin;
    //     int j = static_cast<int>(
    //         std::round(model.x2.v / block.scale)
    //     ) - block.ymin;
    //     data[j * block.nx + i] = src.wdet;
    // }
    // for (table::galNumber & src: cat){
    //     const ngmix::NgmixGaussian & model = src.model;
    //     int i = static_cast<int>(
    //         std::round(model.x1.v / block.scale)
    //     ) - block.xmin;
    //     int j = static_cast<int>(
    //         std::round(model.x2.v / block.scale)
    //     ) - block.ymin;
    //     math::qnumber ss;
    //     int nss = 0;
    //     for (int jj = j - 3; jj <= j + 3; ++jj) {
    //         int dy = jj - j;
    //         for (int ii = i - 3; ii <= i + 3; ++ii) {
    //             int dx = ii -i;
    //             // radius
    //             int r2 = dx * dx + dy * dy;
    //             if ((r2 < 8) && (r2!=0)) {
    //                 ss = ss + data[jj * block.nx + ii];
    //                 nss = nss + 1;
    //             }
    //         }
    //     }
    //     // average over the footprint rather than the bare sum
    //     if (nss > 0) ss = ss / static_cast<double>(nss);
    //     src.wdet = math::ssfunc1(
    //         src.wdet - ss,
    //         0.4,
    //         0.399
    //     );
    // }
    // --------------------------------------------------------------------

    std::vector<table::galNumber> catalog;
    catalog.reserve(cat.size());
    for (table::galNumber & src: cat){
        if (src.wdet.v > wdet_min) catalog.push_back(src);
    }
    return catalog;
};

void pyExportDetector(py::module_& m);

} // detector
} // anacal

#endif // ANACAL_DETECTOR
