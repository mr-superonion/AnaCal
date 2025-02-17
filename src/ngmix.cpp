#include "anacal.h"

namespace anacal {
namespace ngmix {
void
pyExportNgmix(py::module_& m) {
    py::module_ ngmix = m.def_submodule(
        "ngmix", "submodule for ngmix shape, flux estimation"
    );

    py::class_<modelNumber>(ngmix, "modelNumber")
        .def(py::init<
            math::qnumber, math::qnumber, math::qnumber, math::qnumber,
            math::qnumber, math::qnumber
            >(),
            py::arg("A"), py::arg("t"), py::arg("e1"), py::arg("e2"),
            py::arg("x1"), py::arg("x2")
        )
        .def_readwrite("A", &modelNumber::A)
        .def_readwrite("t", &modelNumber::t)
        .def_readwrite("e1", &modelNumber::e1)
        .def_readwrite("e2", &modelNumber::e2)
        .def_readwrite("x1", &modelNumber::x1)
        .def_readwrite("x2", &modelNumber::x2);

    py::class_<lossNumber>(ngmix, "lossNumber")
        .def(py::init<
            math::qnumber, math::qnumber, math::qnumber, math::qnumber,
            math::qnumber, math::qnumber, math::qnumber, math::qnumber,
            math::qnumber, math::qnumber, math::qnumber, math::qnumber,
            math::qnumber>(),
            py::arg("v"),
            py::arg("v_A"), py::arg("v_t"), py::arg("v_e1"), py::arg("v_e2"),
            py::arg("v_x1"), py::arg("v_x2"),
            py::arg("v_AA"), py::arg("v_tt"), py::arg("v_e1e1"),
            py::arg("v_e2e2"), py::arg("v_x1x1"), py::arg("v_x2x2")
        )
        .def_readwrite("v", &lossNumber::v)
        .def_readwrite("v_A", &lossNumber::v_A)
        .def_readwrite("v_t", &lossNumber::v_t)
        .def_readwrite("v_e1", &lossNumber::v_e1)
        .def_readwrite("v_e2", &lossNumber::v_e2)
        .def_readwrite("v_x1", &lossNumber::v_x1)
        .def_readwrite("v_x2", &lossNumber::v_x2)
        .def_readwrite("v_AA", &lossNumber::v_AA)
        .def_readwrite("v_tt", &lossNumber::v_tt)
        .def_readwrite("v_e1e1", &lossNumber::v_e1e1)
        .def_readwrite("v_e2e2", &lossNumber::v_e2e2)
        .def_readwrite("v_x1x1", &lossNumber::v_x1x1)
        .def_readwrite("v_x2x2", &lossNumber::v_x2x2);

    py::class_<galNumber>(ngmix, "galNumber")
        .def(py::init<>())
        .def(py::init<
            modelNumber, math::qnumber, lossNumber
            >(),
            py::arg("params"), py::arg("wedt"), py::arg("loss")
        )
        .def_readwrite("params", &galNumber::params)
        .def_readwrite("wdet", &galNumber::wdet)
        .def_readwrite("loss", &galNumber::loss);

    py::class_<NgmixModel>(ngmix, "NgmixModel")
        .def(py::init<>())
        .def_readonly("det", &NgmixModel::det)
        .def("set_params", &NgmixModel::set_params,
            "Sets the transform parameters",
            py::arg("params")
        )
        .def("prepare_grad", &NgmixModel::prepare_grad,
            "Prepare the gradient function"
        )
        .def("get_r2", &NgmixModel::get_r2,
            "Returns the r squared value at x, y",
            py::arg("x"), py::arg("y")
        )
        .def("model", &NgmixModel::model,
            "Returns the distorted model value at x, y",
            py::arg("x"), py::arg("y")
        )
        .def("loss", &NgmixModel::loss,
            "Returns the loss function value and the derivatives wrt params",
            py::arg("image_val"), py::arg("variance_val"),
            py::arg("x"), py::arg("y")
        );

    py::class_<NgmixGaussian, NgmixModel>(ngmix, "NgmixGaussian")
        .def(py::init<double>(),
            "NgmixGaussian basis function",
            py::arg("sigma_arcsec")
        )
        .def_readwrite("A", &NgmixGaussian::A)
        .def_readwrite("t", &NgmixGaussian::t)
        .def_readwrite("rho", &NgmixGaussian::rho)
        .def_readwrite("e1", &NgmixGaussian::e1)
        .def_readwrite("e2", &NgmixGaussian::e2)
        .def_readwrite("x1", &NgmixGaussian::x1)
        .def_readwrite("x2", &NgmixGaussian::x2)
        .def("get_flux_stamp", &NgmixModel::get_flux_stamp,
            "Returns the flux on a stamp",
            py::arg("nx"), py::arg("ny"), py::arg("scale")
        )
        .def("get_image_stamp", &NgmixModel::get_image_stamp,
            "Returns the flux on a stamp",
            py::arg("nx"), py::arg("ny"), py::arg("scale")
        );

    py::class_<GaussFit>(ngmix, "GaussFit")
        .def(
            py::init<
                int, int, double, double, double, bool,
                const std::optional<modelNumber>,
                const std::optional<modelNumber>,
                int
            >(),
            py::arg("nx"),
            py::arg("ny"),
            py::arg("scale"),
            py::arg("sigma_arcsec"),
            py::arg("klim"),
            py::arg("use_estimate")=true,
            py::arg("prior_mu")=py::none(),
            py::arg("prior_sigma")=py::none(),
            py::arg("stamp_size")=64
        )
        .def_readwrite("model", &GaussFit::model)
        .def("process_block", &GaussFit::process_block,
            "Run iteration for fitting",
            py::arg("catalog"),
            py::arg("img_array"),
            py::arg("psf_array"),
            py::arg("noise_array")=py::none(),
            py::arg("num_epochs")=5,
            py::arg("xcen")=-1,
            py::arg("ycen")=-1,
            py::arg("variance")=1.0
        );
}

} // end of ngmix
} // end of anacal
