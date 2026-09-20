#include "anacal.h"

namespace anacal {
namespace ngmix {
void
pyExportNgmix(py::module_& m) {
    py::module_ ngmix = m.def_submodule(
        "ngmix", "submodule for ngmix shape, flux estimation"
    );

    py::class_<modelPrior>(ngmix, "modelPrior")
        .def(py::init<>())
        .def_readonly("w_F", &modelPrior::w_F)
        .def_readonly("w_a", &modelPrior::w_a)
        .def_readonly("w_x", &modelPrior::w_x)
        .def("set_sigma_F", &modelPrior::set_sigma_F,
            "set the Gaussian prior on Flux",
            py::arg("sigma_F")
        )
        .def("set_sigma_a", &modelPrior::set_sigma_a,
            "set the Gaussian prior (towards 0, width in arcsec^2) on "
            "each intrinsic covariance component mxx, myy, mxy",
            py::arg("sigma_a")
        )
        .def("set_sigma_x", &modelPrior::set_sigma_x,
            "set the Gaussian prior on position",
            py::arg("sigma_x")
        );

    py::class_<modelKernelB>(ngmix, "modelKernelB")
        .def(py::init<>())
        .def_readonly("f", &modelKernelB::f)
        .def_readonly("ixx", &modelKernelB::ixx)
        .def_readonly("ixy", &modelKernelB::ixy)
        .def_readonly("iyy", &modelKernelB::iyy);

    py::class_<modelKernelD>(ngmix, "modelKernelD")
        .def(py::init<>())
        .def_readonly("f", &modelKernelD::f)
        .def_readonly("ixx", &modelKernelD::ixx)
        .def_readonly("ixy", &modelKernelD::ixy)
        .def_readonly("iyy", &modelKernelD::iyy)
        .def_readonly("f_mxx", &modelKernelD::f_mxx)
        .def_readonly("f_myy", &modelKernelD::f_myy)
        .def_readonly("f_mxy", &modelKernelD::f_mxy);

    py::class_<NgmixGaussian>(ngmix, "NgmixGaussian")
        .def(py::init<bool, bool>(),
            py::arg("force_size")=false,
            py::arg("force_center")=false
        )
        .def_readwrite("F", &NgmixGaussian::F)
        .def_readwrite("mxx", &NgmixGaussian::mxx)
        .def_readwrite("myy", &NgmixGaussian::myy)
        .def_readwrite("mxy", &NgmixGaussian::mxy)
        // a1 / a2 / t are derived from the covariance; setting one of
        // them rebuilds the covariance from the (a1, a2, t) triple
        .def_property("a1",
            [](const NgmixGaussian& m) { return m.get_axes()[0]; },
            [](NgmixGaussian& m, const math::qnumber& v) {
                auto ax = m.get_axes(); m.set_axes(v, ax[1], ax[2]);
            }
        )
        .def_property("a2",
            [](const NgmixGaussian& m) { return m.get_axes()[1]; },
            [](NgmixGaussian& m, const math::qnumber& v) {
                auto ax = m.get_axes(); m.set_axes(ax[0], v, ax[2]);
            }
        )
        .def_property("t",
            [](const NgmixGaussian& m) { return m.get_axes()[2]; },
            [](NgmixGaussian& m, const math::qnumber& v) {
                auto ax = m.get_axes(); m.set_axes(ax[0], ax[1], v);
            }
        )
        .def("set_axes", &NgmixGaussian::set_axes,
            "Set the intrinsic covariance from semi-axes a1 (along t), a2 "
            "and the angle t",
            py::arg("a1"), py::arg("a2"), py::arg("t")
        )
        .def("get_axes", &NgmixGaussian::get_axes,
            "Derived (a1, a2, t) of the intrinsic covariance"
        )
        .def_readwrite("x1", &NgmixGaussian::x1)
        .def_readwrite("x2", &NgmixGaussian::x2)
        .def_readwrite("sigma2_shape", &NgmixGaussian::sigma2_shape)
        .def_readwrite("force_size", &NgmixGaussian::force_size)
        .def_readwrite("force_center", &NgmixGaussian::force_center)
        .def("prepare_modelD", &NgmixGaussian::prepare_modelD,
            "Prepare the gradient function",
            py::arg("scale"), py::arg("sigma_arcsec")
        )
        .def("get_r2",
             py::overload_cast<
                 double, double, const modelKernelB&
             >(&NgmixGaussian::get_r2, py::const_),
             "Returns the r squared value at x, y using modelKernelB (as a math.qnumber).",
             py::arg("x"), py::arg("y"), py::arg("c")
        )
        .def("get_r2",
             py::overload_cast<
                 double, double, const modelKernelD&
             >(&NgmixGaussian::get_r2, py::const_),
             "Returns the r squared value and its first-order derivatives at x, y using modelKernelD (as a math.lossNumber).",
             py::arg("x"), py::arg("y"), py::arg("c")
        )
        .def("get_model", &NgmixGaussian::get_model,
            "Returns the distorted model value at x, y",
            py::arg("x"), py::arg("y"), py::arg("c")
        )
        .def("get_loss", &NgmixGaussian::get_loss,
            "chi2 of one pixel with its gradient and Gauss-Newton curvature "
            "(diagonal and cross terms) in (F, mxx, myy, mxy, x1, x2), given "
            "the pixel value, its variance, the r2 lossNumber from get_r2 "
            "and the modelKernelD.",
            py::arg("img_val"), py::arg("variance_val"), py::arg("r2"),
            py::arg("c")
        )
        .def("get_shape", &NgmixGaussian::get_shape,
            "Returns shape (e1, e2)"
        )
        .def("get_flux_stamp", &NgmixGaussian::get_flux_stamp,
            "Returns the flux on a stamp",
            py::arg("nx"), py::arg("ny"), py::arg("scale"), py::arg("sigma_arcsec")
        )
        .def("get_image_stamp", &NgmixGaussian::get_image_stamp,
            "Returns the image on a stamp",
            py::arg("nx"), py::arg("ny"), py::arg("scale"), py::arg("sigma_arcsec")
        );

    py::class_<GaussFit>(ngmix, "GaussFit")
        .def(
            py::init<
                double, double, int, bool, bool, double, bool,
                double, double, double, double, double, double, double
            >(),
            py::arg("scale"),
            py::arg("sigma_arcsec"),
            py::arg("stamp_size")=64,
            py::arg("force_size")=false,
            py::arg("force_center")=false,
            py::arg("fpfs_c0")=1.0,
            py::arg("do_fpfs")=true,
            py::arg("lm_lambda0")=0.2,
            py::arg("lm_decay")=0.5,
            py::arg("damping_floor")=50.0,
            py::arg("conv_tol")=1.0e-3,
            py::arg("trust_shape")=0.05,
            py::arg("trust_center")=0.1,
            py::arg("misfit_damping")=1.0
        )
        .def("process_cell", &GaussFit::process_cell,
            "Run iteration for fitting",
            py::arg("catalog"),
            py::arg("img_array"),
            py::arg("psf_array"),
            py::arg("prior"),
            py::arg("noise_array")=py::none(),
            py::arg("num_epochs")=5,
            py::arg("variance")=1.0,
            py::arg("cell")=py::none(),
            py::arg("n_mask_base_max")=py::none()
        );
}

} // end of ngmix
} // end of anacal
