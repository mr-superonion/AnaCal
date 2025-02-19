#ifndef ANACAL_NGMIX_FITTING
#define ANACAL_NGMIX_FITTING

#include "../image.h"
#include "../math.h"
#include "../stdafx.h"
#include "../table.h"


namespace anacal {
namespace ngmix {

class GaussFit {
public:
    // block dimension
    int nx, ny;
    // stamp dimension
    int stamp_size, ss2;
    double sigma_arcsec;
    double scale;
    math::qnumber w_A, w_t, w_e1, w_e2;
    math::qnumber w_x1, w_x2;
    math::qnumber mu_A, mu_t, mu_e1, mu_e2;
    math::qnumber mu_x1, mu_x2;
    ImageQ image;
    NgmixGaussian model;

    GaussFit(
        int nx,
        int ny,
        double scale,
        double sigma_arcsec,
        double klim,
        bool use_estimate = true,
        const std::optional<modelNumber> & prior_mu = std::nullopt,
        const std::optional<modelNumber> & prior_sigma = std::nullopt,
        int stamp_size=64
    ) : image(nx, ny, scale, sigma_arcsec, klim, use_estimate),
        model(sigma_arcsec)
    {
        this->nx = nx;
        this->ny = ny;
        if ((nx % 2 != 0) || (ny % 2 != 0)) {
            throw std::invalid_argument("nx or ny is not even number");
        }
        this->stamp_size = stamp_size;
        this->ss2 = stamp_size / 2;
        this->sigma_arcsec = sigma_arcsec;
        this->scale = scale;
        if (sigma_arcsec <= 0) {
            throw std::invalid_argument("sigma_arcsec must be positive");
        }
        // If prior_mu is provided, assign its values
        if (prior_mu) {
            this->mu_A = (*prior_mu).A;
            this->mu_t = (*prior_mu).t;
            this->mu_e1 = (*prior_mu).e1;
            this->mu_e2 = (*prior_mu).e2;
            this->mu_x1 = (*prior_mu).x1;
            this->mu_x2 = (*prior_mu).x2;
        }

        // If prior_sigma is provided, assign its values
        if (prior_sigma) {
            if ((*prior_sigma).A.v > 0.0){
                this->w_A = 2.0 / math::pow((*prior_sigma).A, 2.0);
            } else {
                this->w_A = {0.0, 0.0, 0.0, 0.0, 0.0};
            }

            if ((*prior_sigma).t.v > 0.0){
                this->w_t = 2.0 / math::pow((*prior_sigma).t, 2.0);
            } else {
                this->w_t = {0.0, 0.0, 0.0, 0.0, 0.0};
            }

            if ((*prior_sigma).e1.v > 0.0){
                this->w_e1 = 2.0 / math::pow((*prior_sigma).e1, 2.0);
            } else {
                this->w_e1 = {0.0, 0.0, 0.0, 0.0, 0.0};
            }

            if ((*prior_sigma).e2.v > 0.0){
                this->w_e2 = 2.0 / math::pow((*prior_sigma).e2, 2.0);
            } else {
                this->w_e2 = {0.0, 0.0, 0.0, 0.0, 0.0};
            }

            if ((*prior_sigma).x1.v > 0.0){
                this->w_x1 = 2.0 / math::pow((*prior_sigma).x1, 2.0);
            } else{
                this->w_x1 = {0.0, 0.0, 0.0, 0.0, 0.0};
            }

            if ((*prior_sigma).x2.v > 0.0){
                this->w_x2 = 2.0 / math::pow((*prior_sigma).x2, 2.0);
            } else {
                this->w_x2 = {0.0, 0.0, 0.0, 0.0, 0.0};
            }
        }

    };

    void update_model_params(
        const lossNumber& loss
    ) {
        this->model.A = this->model.A - (
            (loss.v_A + this->w_A * (this->model.A - this->mu_A)) / (
                loss.v_AA + this->w_A
            )
        );
        this->model.t = this->model.t - (
            (loss.v_t + this->w_t * (this->model.t - this->mu_t)) / (
                loss.v_tt + this->w_t
            )
        );
        this->model.rho = math::exp(this->model.t);
        this->model.e1 = this->model.e1 - (
            (loss.v_e1 + this->w_e1 * (this->model.e1 - this->mu_e1)) / (
                loss.v_e1e1 + this->w_e1
            )
        );
        this->model.e2 = this->model.e2 - (
            (loss.v_e2 + this->w_e2 * (this->model.e2 - this->mu_e2)) / (
                loss.v_e2e2 + this->w_e2
            )
        );
        this->model.x1 = this->model.x1 - (
            (loss.v_x1 + this->w_x1 * (this->model.x1 - this->mu_x1)) / (
                loss.v_x1x1 + this->w_x1
            )
        );
        this->model.x2 = this->model.x2 - (
            (loss.v_x2 + this->w_x2 * (this->model.x2 - this->mu_x2)) / (
                loss.v_x2x2 + this->w_x2
            )
        );
    };

    lossNumber accumulate_loss(
        const std::vector<math::qnumber> & data,
        int x_stamp,
        int y_stamp,
        double variance=1.0
    ) {
        lossNumber loss;
        for (int j = 0; j < this->stamp_size; ++j) {
            int jj = j + y_stamp - this->ss2;
            if (jj < 0 || jj >= this->ny) {
                continue;
            }
            double y = (j - this->ss2) * this->scale;
            for (int i = 0; i < this->stamp_size; ++i) {
                int ii = i + x_stamp - this->ss2;
                if (ii < 0 || ii >= this->nx) {
                    continue;
                }
                double x = (i - this->ss2) * this->scale;
                int index = jj * this->nx + ii;
                loss = loss + this->model.loss(
                    data[index], variance, x, y
                );
            }
        }
        return loss;
    };

    std::vector<table::galNumber> process_block(
        const std::vector<table::galNumber> & catalog,
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt,
        int num_epochs = 5,
        double variance = 1.0,
        std::optional<table::block> block=std::nullopt
    ) {

        int arr_ny = img_array.shape(0);
        int arr_nx = img_array.shape(1);
        table::block bb;
        if (block) {
            bb = *block;
        } else {
            bb = table::get_block_center_list(
                arr_nx, arr_ny, arr_nx, arr_ny, 0
            )[0];
        }
        std::vector<math::qnumber> data = this->image.prepare_qnumber_vector(
            img_array,
            psf_array,
            noise_array,
            bb.xcen,
            bb.ycen
        );
        size_t ngal = catalog.size();
        std::vector<table::galNumber> result(ngal);
        for (size_t i = 0; i < ngal; i++) {
            table::galNumber src = catalog[i];
            // Iterative optimization loop
            for (int epoch = 0; epoch < num_epochs; ++epoch) {
                int x_stamp = static_cast<int>(
                    std::round(src.params.x1.v / this->scale)
                );
                int y_stamp = static_cast<int>(
                    std::round(src.params.x2.v / this->scale)
                );
                src.params.x1.v = src.params.x1.v - x_stamp * this->scale;
                src.params.x2.v = src.params.x2.v - y_stamp * this->scale;
                this->model.set_params(src.params);
                this->model.prepare_grad();
                src.loss = this->accumulate_loss(
                    data, x_stamp, y_stamp, variance
                );
                this->update_model_params(src.loss);
                src.params = this->model.get_params();
                src.params.x1.v = src.params.x1.v + x_stamp * this->scale;
                src.params.x2.v = src.params.x2.v + y_stamp * this->scale;
            }
            result[i] = src;
        }
        return result;
    };
};

} // end of ngmix
} // end of anacal

#endif // ANACAL_NGMIX_FITTING
