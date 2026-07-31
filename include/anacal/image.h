#ifndef ANACAL_IMAGE_H
#define ANACAL_IMAGE_H

#include "table.h"
#include "model.h"
#include "math/tensor.h"
#include <fftw3.h>
#include <utility>

namespace anacal {
inline constexpr double min_deconv_ratio = 1e-5;

class Image {
private:
    fftw_plan plan_forward = nullptr;
    fftw_plan plan_backward = nullptr;
    int nx2, ny2, npixels, npixels_f;
    int kx_length, ky_length;
    double dkx, dky;
    double norm_factor;
    double* data_r = nullptr;
    fftw_complex* data_f = nullptr;

    // Preventing copy (implement these if you need copy semantics)
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    unsigned int mode;

    void _rotate90_f(int);

    void assert_mode(unsigned int condition) const {
        if (condition==0) {
            throw std::runtime_error(
                "Error: The Image object has a wrong mode."
            );
        }
    }
public:
    int ny, nx;
    double scale=1;

    Image(
        int nx,
        int ny,
        double scale,
        bool use_estimate=true,
        unsigned int mode=3
    );

    // Templated on the input pixel type so that a float32 science or noise
    // plane can be read without first making a float64 copy of it.  This is
    // the single point where the pixel type widens: everything downstream of
    // ``data_r`` is double.  Instantiated for float and double in image.cpp.
    template <typename T>
    void set_r(
        const py::array_t<T>&,
        int xcen, int,
        bool ishift=false
    );

    template <typename T>
    void set_r(
        const py::array_t<T>&,
        bool ishift=false
    );

    void set_delta_r(bool ishift=false);

    void set_f(const py::array_t<std::complex<double>>&);

    void set_delta_f();

    void set_noise_f(unsigned int, const py::array_t<double>&);

    void fft();

    void ifft();

    void add_image_f(const py::array_t<std::complex<double>>&);

    void subtract_image_f(const py::array_t<std::complex<double>>&);

    void filter(const py::array_t<std::complex<double>>&);

    void filter(const BaseModel&);

    py::array_t<double> measure(
        const py::array_t<std::complex<double>>&,
        double dy=0.0,
        double dx=0.0
    ) const;

    void deconvolve(
        const py::array_t<std::complex<double>>&,
        double
    );

    void deconvolve(const BaseModel&, double);

    void rotate90_f();

    void irotate90_f();

    py::array_t<std::complex<double>> draw_f() const;

    py::array_t<double> draw_r(bool ishift=false) const;

    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    ~Image();

    inline void truncate(double xlim, bool ishift) {
        int off_x = ishift ? this->nx2 : 0;
        int off_y = ishift ? this->ny2 : 0;
        double xlim2 = xlim * xlim;

        for (int j = 0; j < this->ny; ++j) {
            int jj = ((j + off_y) % this->ny - this->ny2);
            double y = jj * this->scale;

            for (int i = 0; i < this->nx; ++i) {
                int ii = ((i + off_x) % this->nx - this->nx2);
                double x = ii * this->scale;

                double r2 = x * x + y * y;
                int index = j * this->nx + i;

                if (r2 > xlim2) {
                    this->data_r[index] = 0.0;
                }
            }
        }
        return;
    };

    inline py::array_t<std::complex<double>>
    get_lens_kernel(
        const py::array_t<double>& psf_array,
        const Gaussian & gauss_model,
        double klim
    ) {
        // Prepare PSF
        this->set_r(psf_array, true);
        this->fft();
        const py::array_t<std::complex<double>> parr = this->draw_f();
        // Delta
        this->set_delta_f();
        // Convolve Gaussian
        this->filter(gauss_model);
        // Deconvolve the PSF
        this->deconvolve(parr, klim);
        /* this->ifft(); */
        /* // We truncate the deconvolved Gaussian kernel */
        /* this->truncate(xlim, true); */
        /* this->fft(); */
        return this->draw_f();
    };


};

py::array_t<std::complex<double>>
compute_fft(
    int nx,
    int ny,
    const py::array_t<double>& data_in,
    bool ishift=false
);

py::array_t<std::complex<double>>
deconvolve_filter(
    const py::array_t<std::complex<double>>& filter_image,
    const py::array_t<std::complex<double>>& parr,
    double scale,
    double klim
);


class ImageQ {
private:
    // Preventing copy (implement these if you need copy semantics)
    double klim;
    int nx, ny;
    double scale;
    bool use_estimate;
    Image img_obj;
    double sigma_arcsec;

    const Gaussian gauss_model;
    const GaussianG1 gauss_g1_model;
    const GaussianG2 gauss_g2_model;
    const GaussianX1 gauss_x1_model;
    const GaussianX2 gauss_x2_model;

    int nx2, ny2;
    int kx_length, ky_length;
    ImageQ(const ImageQ&) = delete;
    ImageQ& operator=(const ImageQ&) = delete;
public:
    ImageQ(
        int nx,
        int ny,
        double scale,
        double sigma_arcsec,
        double klim,
        bool use_estimate = true
    ) :
        nx(nx), ny(ny), scale(scale), use_estimate(use_estimate),
        img_obj(nx, ny, scale, use_estimate),
        sigma_arcsec(sigma_arcsec),
        gauss_model(1.0 / sigma_arcsec),
        gauss_g1_model(1.0 / sigma_arcsec),
        gauss_g2_model(1.0 / sigma_arcsec)

    {
        if ((sigma_arcsec <= 0) || (sigma_arcsec > 5.0)) {
            throw std::runtime_error(
                "ImageQ Error: invalid input sigma_arcsec"
            );
        }
        this->nx2 = nx / 2;
        this->ny2 = ny / 2;
        this->kx_length = nx / 2 + 1;
        this->ky_length = ny;
        this->klim = klim;
        return;
    };

    std::vector<math::qnumber>
    prepare_qnumber_vector(
        const py::array_t<pixel_t>& img_array,
        const py::array_t<double>& psf_array,
        int xcen,
        int ycen,
        const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt
    ) {
        const py::array_t<std::complex<double>> karr = img_obj.get_lens_kernel(
            psf_array,
            gauss_model,
            klim
        );
        // signal
        img_obj.set_r(img_array, xcen, ycen, false);
        img_obj.fft();
        img_obj.filter(karr);
        py::array_t<std::complex<double>> imgcov_f = img_obj.draw_f();

        // image is derived from image + noise
        py::array_t<std::complex<double>> imgcov_f_n;
        bool has_noise = noise_array.has_value();
        if (has_noise) {
            img_obj.set_f(karr);
            img_obj.rotate90_f();
            const py::array_t<std::complex<double>> karr_n = img_obj.draw_f();

            // signal
            img_obj.set_r(*noise_array, xcen, ycen, false);
            img_obj.fft();
            img_obj.filter(karr_n);         // Filtering
            imgcov_f_n = img_obj.draw_f();

            auto r = imgcov_f.mutable_unchecked<2>();
            auto r_n = imgcov_f_n.unchecked<2>();
            for (int j = 0; j < this->ky_length ; ++j) {
                for (int i = 0; i < this->kx_length ; ++i) {
                    r(j, i) = r(j, i) + r_n(j, i);
                }
            }
        }

        std::vector<math::qnumber> result(this->ny * this->nx);
        // v
        {
            img_obj.set_f(imgcov_f);
            img_obj.ifft();
            auto tmp_r = img_obj.draw_r().unchecked<2>();
            for (ssize_t j = 0, idx = 0; j < this->ny; ++j) {
                for (ssize_t i = 0; i < this->nx; ++i, ++idx) {
                    result[idx].v = tmp_r(j, i);
                }
            }
        }

        // shear response from image - noise
        if (has_noise) {
            auto r = imgcov_f.mutable_unchecked<2>();
            auto r_n = imgcov_f_n.unchecked<2>();
            for (int j = 0; j < this->ky_length ; ++j) {
                for (int i = 0; i < this->kx_length ; ++i) {
                    r(j, i) = r(j, i) - 2.0 * r_n(j, i);
                }
            }
        }

        // g1
        {
            img_obj.set_f(imgcov_f);
            img_obj.filter(gauss_g1_model);
            img_obj.ifft();
            auto tmp_r = img_obj.draw_r().unchecked<2>();
            for (ssize_t j = 0, idx=0; j < this->ny; ++j) {
                for (ssize_t i = 0; i < this->nx; ++i, ++idx) {
                    result[idx].g1 = tmp_r(j, i);
                }
            }
        }

        // g2
        {
            img_obj.set_f(imgcov_f);
            img_obj.filter(gauss_g2_model);
            img_obj.ifft();
            auto tmp_r = img_obj.draw_r().unchecked<2>();
            for (ssize_t j = 0, idx=0; j < this->ny; ++j) {
                for (ssize_t i = 0; i < this->nx; ++i, ++idx) {
                    result[idx].g2 = tmp_r(j, i);
                }
            }
        }

        // x1
        {
            img_obj.set_f(imgcov_f);
            img_obj.filter(gauss_x1_model);
            img_obj.ifft();
            py::array_t<double> tmp = img_obj.draw_r();
            auto tmp_r = tmp.unchecked<2>();
            for (ssize_t j = 0, idx=0; j < this->ny; ++j) {
                double y = (j - this->ny2) * this->scale;
                for (ssize_t i = 0; i < this->nx; ++i, ++idx) {
                    double x = (i - this->nx2) * this->scale;
                    result[idx].g1 = result[idx].g1 + x * tmp_r(j, i);
                    result[idx].g2 = result[idx].g2 + y * tmp_r(j, i);
                    result[idx].x1 = tmp_r(j, i);
                }
            }
        }

         // x2
        {
            img_obj.set_f(imgcov_f);
            img_obj.filter(gauss_x2_model);
            img_obj.ifft();
            py::array_t<double> tmp = img_obj.draw_r();
            auto tmp_r = tmp.unchecked<2>();
            for (ssize_t j = 0, idx=0; j < this->ny; ++j) {
                double y = (j - this->ny2) * this->scale;
                for (ssize_t i = 0; i < this->nx; ++i, ++idx) {
                    double x = (i - this->nx2) * this->scale;
                    result[idx].g1 = result[idx].g1 - y * tmp_r(j, i);
                    result[idx].g2 = result[idx].g2 + x * tmp_r(j, i);
                    result[idx].x2 = tmp_r(j, i);
                }
            }
        }
        return result;
    };

    math::qtensor
    prepare_qtensor(
        const py::array_t<pixel_t>& img_array,
        const py::array_t<double>& psf_array,
        int xcen,
        int ycen,
        const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt
    ) {
        auto data = prepare_qnumber_vector(
            img_array,
            psf_array,
            xcen,
            ycen,
            noise_array
        );
        return math::qtensor::from_image(
            std::move(data),
            static_cast<std::size_t>(this->ny),
            static_cast<std::size_t>(this->nx)
        );
    };

    py::array_t<double>
    prepare_qnumber_image(
        const py::array_t<pixel_t>& img_array,
        const py::array_t<double>& psf_array,
        int xcen,
        int ycen,
        const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt
    ) {
        auto result = py::array_t<double>({5, ny, nx});
        auto r = result.mutable_unchecked<3>();

        std::vector<math::qnumber> qvec = prepare_qnumber_vector(
            img_array,
            psf_array,
            xcen,
            ycen,
            noise_array
        );
        for (ssize_t j = 0, idx=0; j < this->ny; ++j) {
            for (ssize_t i = 0; i < this->nx; ++i, ++idx) {
                r(0, j, i) = qvec[idx].v;
                r(1, j, i) = qvec[idx].g1;
                r(2, j, i) = qvec[idx].g2;
                r(3, j, i) = qvec[idx].x1;
                r(4, j, i) = qvec[idx].x2;
            }
        }
        return result;
    };


    ImageQ(ImageQ&& other) noexcept = default;
    ImageQ& operator=(ImageQ&& other) noexcept = default;

    ~ImageQ() = default;
};

// ---------------------------------------------------------------------------
// Multi-band inputs
//
// Detection may be run on several bands at once.  An image, PSF or noise
// argument is then a stack: (nband, ny, nx) instead of (ny, nx).  A plain 2-D
// array is simply the nband = 1 case, so there is one code path, not two.
// ---------------------------------------------------------------------------

// Noise variance as a caller may give it: one number for a plain image, or one
// number per band for a stack.  Normalised to a vector on the way in, so the
// rest of the code only ever sees the vector form.
using varianceArg = std::variant<double, std::vector<double>>;

inline std::vector<double>
to_variance_vector(const varianceArg& variance) {
    if (std::holds_alternative<double>(variance)) {
        return std::vector<double>{std::get<double>(variance)};
    }
    return std::get<std::vector<double>>(variance);
};

// Number of bands an argument carries: 1 for a 2-D array, shape(0) for a 3-D
// stack.
template <typename T>
inline ssize_t
nband_of(const py::array_t<T>& arr) {
    if (arr.ndim() == 2) return 1;
    if (arr.ndim() == 3) return arr.shape(0);
    throw std::runtime_error(
        "anacal Error: expected a 2-D image or a 3-D (nband, ny, nx) stack, "
        "got " + std::to_string(arr.ndim()) + " dimensions"
    );
};

// One 2-D view onto band ``b``.  No pixels are copied: the stack is handed in
// as the base object so it stays alive, and the view keeps the stack's strides,
// which ``Image::set_r`` honours because it reads through ``unchecked<2>()``.
template <typename T>
inline py::array_t<T>
band_view(const py::array_t<T>& stack, ssize_t b) {
    if (stack.ndim() == 2) {
        if (b != 0) {
            throw std::runtime_error(
                "anacal Error: band index out of range for a 2-D image"
            );
        }
        return stack;
    }
    if (b < 0 || b >= stack.shape(0)) {
        throw std::runtime_error("anacal Error: band index out of range");
    }
    const T* ptr = reinterpret_cast<const T*>(
        reinterpret_cast<const char*>(stack.data()) + b * stack.strides(0)
    );
    return py::array_t<T>(
        {stack.shape(1), stack.shape(2)},
        {stack.strides(1), stack.strides(2)},
        ptr,
        stack
    );
};

// Check that image, PSF, noise and variance all describe the same set of
// bands, and return how many there are.  A band whose variance is not a
// positive finite number is an error rather than something to drop silently:
// dropping it would quietly change the depth of the coadd.
inline ssize_t
check_band_stack(
    const py::array_t<pixel_t>& img_stack,
    const py::array_t<double>& psf_stack,
    const std::vector<double>& variance,
    const std::optional<py::array_t<pixel_t>>& noise_stack=std::nullopt
) {
    const ssize_t nband = nband_of(img_stack);
    if (nband_of(psf_stack) != nband) {
        throw std::runtime_error(
            "anacal Error: image has " + std::to_string(nband) +
            " band(s) but the PSF has " +
            std::to_string(nband_of(psf_stack))
        );
    }
    if (static_cast<ssize_t>(variance.size()) != nband) {
        throw std::runtime_error(
            "anacal Error: image has " + std::to_string(nband) +
            " band(s) but " + std::to_string(variance.size()) +
            " variance value(s) were given"
        );
    }
    if (noise_stack.has_value() && (nband_of(*noise_stack) != nband)) {
        throw std::runtime_error(
            "anacal Error: image has " + std::to_string(nband) +
            " band(s) but the noise image has " +
            std::to_string(nband_of(*noise_stack))
        );
    }
    for (ssize_t b = 0; b < nband; ++b) {
        const double v = variance[b];
        if (!std::isfinite(v) || (v <= 0.0)) {
            throw std::runtime_error(
                "anacal Error: variance of band " + std::to_string(b) +
                " is " + std::to_string(v) + "; it must be positive and finite"
            );
        }
    }
    return nband;
};

inline std::vector<math::qnumber>
prepare_data_block(
    const py::array_t<pixel_t>& img_array,
    const py::array_t<double>& psf_array,
    double sigma_arcsec,
    const geometry::block & block,
    const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt
){
    ImageQ img_obj(
        block.nx,
        block.ny,
        block.scale,
        sigma_arcsec,
        3.1 / block.scale,      // klim = 3.1 / scale (matches
                                // gaussian_flux_variance in task.h)
        true                    // us estimate in FFTW
    );
    return img_obj.prepare_qnumber_vector(
        img_array,
        psf_array,
        block.xcen,
        block.ycen,
        noise_array
    );
};

inline py::array_t<double>
prepare_data_block_image(
    const py::array_t<pixel_t>& img_array,
    const py::array_t<double>& psf_array,
    double sigma_arcsec,
    const geometry::block & block,
    const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt
) {
    auto result = py::array_t<double>({5, block.ny, block.nx});
    auto r = result.mutable_unchecked<3>();

    std::vector<math::qnumber> qvec = prepare_data_block(
        img_array, psf_array, sigma_arcsec, block, noise_array
    );
    for (ssize_t j = 0, idx=0; j < block.ny; ++j) {
        for (ssize_t i = 0; i < block.nx; ++i, ++idx) {
            r(0, j, i) = qvec[idx].v;
            r(1, j, i) = qvec[idx].g1;
            r(2, j, i) = qvec[idx].g2;
            r(3, j, i) = qvec[idx].x1;
            r(4, j, i) = qvec[idx].x2;
        }
    }
    return result;
};


inline std::vector<math::qnumber>
prepare_model_block(
    const py::array_t<table::galRow> catalog,
    double sigma_arcsec,
    const geometry::block & block
){
    std::vector<table::galNumber> cat = table::array_to_objlist(
        catalog
    );
    std::size_t n_pix = block.nx * block.ny;
    std::vector<math::qnumber> data_model(n_pix);
    for (const table::galNumber& ss : cat) {
        const ngmix::modelKernelB kernel = ss.model.prepare_modelB(
            block.scale,
            sigma_arcsec
        );
        ss.model.add_to_block(data_model, block, kernel);
    }
    return data_model;
};

inline py::array_t<double>
prepare_model_block_image(
    const py::array_t<table::galRow> catalog,
    double sigma_arcsec,
    const geometry::block & block
) {
    auto result = py::array_t<double>({5, block.ny, block.nx});
    auto r = result.mutable_unchecked<3>();

    std::vector<math::qnumber> qvec = prepare_model_block(
        catalog, sigma_arcsec, block
    );
    for (ssize_t j = 0, idx=0; j < block.ny; ++j) {
        for (ssize_t i = 0; i < block.nx; ++i, ++idx) {
            r(0, j, i) = qvec[idx].v;
            r(1, j, i) = qvec[idx].g1;
            r(2, j, i) = qvec[idx].g2;
            r(3, j, i) = qvec[idx].x1;
            r(4, j, i) = qvec[idx].x2;
        }
    }
    return result;
};


inline double get_smoothed_variance(
    double scale,
    double sigma_arcsec,
    const py::array_t<double>& psf_array,
    double variance
) {
    double variance_sm = 0.0;
    // number of pixels in x and y used to estimated noise variance
    // result is independent on this
    int npix = 64;
    Image img_obj(npix, npix, scale, true);
    {
        // Prepare PSF
        img_obj.set_r(psf_array, true);
        img_obj.fft();
        const py::array_t<std::complex<double>> parr = img_obj.draw_f();
        {
            // white noise
            auto pf = py::array_t<std::complex<double>>({npix, npix / 2 + 1});
            auto pf_r = pf.mutable_unchecked<2>();
            std::complex<double> vv(std::sqrt(variance) / npix, 0.0);
            std::complex<double> sqrt2vv = std::sqrt(2.0) * vv;
            for (ssize_t j = 0; j < npix; ++j) {
                for (ssize_t i = 1; i < npix / 2; ++i) {
                    pf_r(j, i) = sqrt2vv;
                }
                pf_r(j, 0) = vv;
                pf_r(j, npix / 2) = vv;
            }
            img_obj.set_f(pf);
        }
        // Deconvolve the PSF
        img_obj.deconvolve(parr, 3.1 / scale);
    }
    {
        // Convolve Gaussian
        const Gaussian gauss_model(1.0 / sigma_arcsec);
        img_obj.filter(gauss_model);
    }
    {
        const py::array_t<std::complex<double>> pf_dec = img_obj.draw_f();
        auto pfd_r = pf_dec.unchecked<2>();
        for (ssize_t j = 0; j < npix; ++j) {
            for (ssize_t i = 0; i < npix / 2 + 1; ++i) {
                variance_sm += std::norm(pfd_r(j, i));
            }
        }
    }
    return variance_sm;
};


// ---------------------------------------------------------------------------
// Combining bands
//
// ``prepare_qnumber_vector`` deconvolves each band's own PSF and reconvolves
// with a Gaussian of width ``sigma_arcsec``, so every band's qimage carries the
// SAME effective PSF.  That is what makes a cross-band average meaningful --
// averaging the raw pixels would not be, because the PSFs differ.
//
// With per-band qimage variance V_b(sigma) and weights w_b summing to one,
//
//     data(sigma)   = sum_b w_b * data_b(sigma)
//     V_coadd(sigma) = sum_b w_b^2 * V_b(sigma)
//
// The weights are chosen ONCE, at the detection smoothing scale, and the same
// set is reused for the measurement.  Detection and measurement then see the
// same coadd; re-deriving the weights per scale would weight the bands -- and
// so the effective colour of every object -- slightly differently at the two
// stages.
//
// For a single band w = {1}: the average is the band itself and V_coadd = V_1,
// bit for bit, so nothing about the one-band results changes.
// ---------------------------------------------------------------------------

// Normalised inverse-variance weights at the given smoothing scale.
inline std::vector<double>
band_weights(
    double scale,
    double sigma_arcsec,
    const py::array_t<double>& psf_stack,
    const std::vector<double>& variance
) {
    const std::size_t nband = variance.size();
    if (nband == 0) {
        throw std::runtime_error("anacal Error: no bands given");
    }
    std::vector<double> w(nband);
    double total = 0.0;
    for (std::size_t b = 0; b < nband; ++b) {
        const double v = get_smoothed_variance(
            scale,
            sigma_arcsec,
            band_view(psf_stack, static_cast<ssize_t>(b)),
            variance[b]
        );
        if (!std::isfinite(v) || (v <= 0.0)) {
            throw std::runtime_error(
                "anacal Error: smoothed variance of band " +
                std::to_string(b) + " is " + std::to_string(v) +
                "; check the PSF stamp and the noise variance"
            );
        }
        w[b] = 1.0 / v;
        total += w[b];
    }
    for (double& x : w) {
        x = x / total;
    }
    return w;
};

// Variance of the coadd at ``sigma_arcsec`` for a fixed set of weights.
inline double
coadd_smoothed_variance(
    double scale,
    double sigma_arcsec,
    const py::array_t<double>& psf_stack,
    const std::vector<double>& variance,
    const std::vector<double>& w
) {
    double out = 0.0;
    for (std::size_t b = 0; b < w.size(); ++b) {
        out = out + w[b] * w[b] * get_smoothed_variance(
            scale,
            sigma_arcsec,
            band_view(psf_stack, static_cast<ssize_t>(b)),
            variance[b]
        );
    }
    return out;
};

// The coadded qimage for one block.  Bands are accumulated one at a time, so
// only two qimages are ever held at once (about 5 MB for a 250 x 250 block)
// rather than one per band.
inline std::vector<math::qnumber>
prepare_data_block_coadd(
    const py::array_t<pixel_t>& img_stack,
    const py::array_t<double>& psf_stack,
    double sigma_arcsec,
    const geometry::block & block,
    const std::vector<double>& w,
    const std::optional<py::array_t<pixel_t>>& noise_stack=std::nullopt
) {
    const std::size_t nband = w.size();
    std::vector<math::qnumber> out;
    for (std::size_t b = 0; b < nband; ++b) {
        const ssize_t ib = static_cast<ssize_t>(b);
        std::optional<py::array_t<pixel_t>> noise_b = std::nullopt;
        if (noise_stack.has_value()) {
            noise_b = band_view(*noise_stack, ib);
        }
        std::vector<math::qnumber> data = prepare_data_block(
            band_view(img_stack, ib),
            band_view(psf_stack, ib),
            sigma_arcsec,
            block,
            noise_b
        );
        if (nband == 1) {
            // One band: return it untouched rather than multiplying by a
            // weight that is exactly 1.0, so the result is bit-identical to
            // the single-band call.
            return data;
        }
        if (b == 0) {
            out = std::move(data);
            for (math::qnumber& q : out) {
                q = q * w[0];
            }
        } else {
            for (std::size_t i = 0; i < out.size(); ++i) {
                out[i] = out[i] + data[i] * w[b];
            }
        }
    }
    return out;
};


void pyExportImage(py::module& m);
}

#endif // IMAGE_H
