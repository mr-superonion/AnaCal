#ifndef ANACAL_NGMIX_FITTING
#define ANACAL_NGMIX_FITTING

#include "../image.h"
#include "../math.h"
#include "../mask.h"
#include "../stdafx.h"


namespace anacal {
namespace ngmix {

class GaussFit {
public:
    int nx, ny;
    double nx2, ny2;
    double sigma_arcsec;
    double scale;
    ImageQ image;
    NgmixGaussian model;

    GaussFit(
        int nx,
        int ny,
        double scale,
        double sigma_arcsec,
        double klim,
        bool use_estimate = true
    ) : image(nx, ny, scale, sigma_arcsec, klim, use_estimate, true),
        model(sigma_arcsec)
    {
        this->nx = nx;
        this->ny = ny;
        this->nx2 = nx / 2;
        this->ny2 = ny / 2;
        this->sigma_arcsec = sigma_arcsec;
        this->scale = scale;
        if (sigma_arcsec <= 0) {
            throw std::invalid_argument("sigma_arcsec must be positive");
        }

    };

    modelNumber get_loss(
        const std::vector<math::qnumber>& data
    ) {

        modelNumber loss;
        for (int j = 0; j < this->ny; ++j) {
            double y = (j - this->ny2) * this->scale;
            for (int i = 0; i < this->nx; ++i) {
                int index = j * this->ny + i;
                double x = (i - this->nx2) * this->scale;
                loss = loss + this->model.loss(
                    data[index], 1.0, x, y
                );
            }
        }
        return loss;

    }

    void update_model_params(
        const modelNumber& loss
    ) {
        this->model.A = this->model.A - (
            loss.v_A / loss.v_AA / 2.0
        );
        this->model.rho = this->model.rho - (
            loss.v_rho / loss.v_rhorho / 2.0
        );
        this->model.Gamma1 = this->model.Gamma1 - (
            loss.v_g1 / loss.v_g1g1 / 2.0
        );
        this->model.Gamma2 = this->model.Gamma2 - (
            loss.v_g2 / loss.v_g2g2 / 2.0
        );
        this->model.x0 = this->model.x0 - (
            loss.v_x  / loss.v_xx / 2.0
        );
        this->model.y0 = this->model.y0 - (
            loss.v_y  / loss.v_yy / 2.0
        );
    };

    void run(
        const std::array<math::qnumber, 6> & params0,
        int xcen,
        int ycen,
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt,
        int num_epochs = 5
    ) {
        std::vector<math::qnumber> data = this->image.prepare_qnumber_vector(
            img_array,
            psf_array,
            noise_array,
            xcen,
            ycen
        );
        this->model.set_params(params0);
        // Iterative optimization loop
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            modelNumber loss = this->get_loss(data);
            this->update_model_params(loss);
        }
    };
};

} // end of ngmix
} // end of anacal

#endif // ANACAL_NGMIX_FITTING
