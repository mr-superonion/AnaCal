#ifndef ANACAL_NGMIX_FITTING
#define ANACAL_NGMIX_FITTING

#include "../image.h"
#include "../math.h"
#include "../mask.h"
#include "../stdafx.h"


namespace anacal {
namespace ngmix {
class Fitting {
public:
    Fitting(
        int nx,
        int ny,
        double sigma,
        int num_epochs = 5,
        double learning_rate = 0.5
    ) : nx(nx),
        ny(ny),
        sigma(sigma),
        num_epochs(num_epochs),
        learning_rate(learning_rate)
    {
        if (sigma <= 0) {
            throw std::invalid_argument("Sigma must be positive");
        }

    };

    double
    loss(const py::array_t<double>& pars)
    {
        this->prepare(pars);
        auto res = this->residual(pars);
        double result = 0.0;
        for (const auto& val : res) {
            result += std::norm(val);
        }
        return result / 2.0;
    }

    py::array_t<double> run(
        const std::vector<math::qnumber>& data,
        const py::array_t<double> & params0
    ) {
        auto p0_r = params0.unchecked<1>();
        int npar = p_r0.shape(0);
        py::array_t<double> params({npar}); // Create a 1D array
        auto p_r = params.mutable_unchecked<1>();
        for (int i = 0; i < npar; ++i) {
            p_r(i) = p0_r(i);
        }

        // Iterative optimization loop
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // Prepare model
            this->prepare(params);

            // Compute gradients and Hessian diagonal
            py::array_t<double> grads = this->dloss(params);
            py::array_t<double> hessian_diag = this->d2loss(params);

            auto g_r = grads.unchecked<1>();
            auto h_r = hessian_diag.unchecked<1>();

            // Update parameters
            for (int i = 0; i < npar; ++i) {
                p_r(i) -= (g_r(i) / h_r(i)) * learning_rate;
            }
        }

        // Return optimized parameters
        return params;
    }

private:
    std::function<double(double)> model_obj;
    int nx, xy;
    int num_epochs;
    double learning_rate;
    double sigma;
    NgmixModel rmodel;
};

}
}

#endif // ANACAL_NGMIX_FITTING
