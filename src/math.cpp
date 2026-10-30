#include "anacal.h"

namespace anacal {
namespace math {
namespace {

using darray = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::vector<double> to_vec(const darray& a) {
    return std::vector<double>(a.data(), a.data() + a.size());
}

std::vector<double> to_vec(const std::optional<darray>& a) {
    return a ? to_vec(*a) : std::vector<double>{};
}

std::size_t dim(const darray& a, py::ssize_t i) {
    return static_cast<std::size_t>(a.shape(i));
}

// [..., 5] numpy array <-> qtensor of shape [...]
qtensor qtensor_from_array(const darray& a) {
    if (a.ndim() < 2 || a.shape(a.ndim() - 1) != 5) {
        throw std::invalid_argument(
            "qtensor.from_array expects an array [..., 5]"
        );
    }
    std::vector<std::size_t> shape;
    for (py::ssize_t i = 0; i + 1 < a.ndim(); ++i) {
        shape.push_back(dim(a, i));
    }
    const std::size_t n = static_cast<std::size_t>(a.size() / 5);
    std::vector<qnumber> data(n);
    const double* p = a.data();
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = qnumber(
            p[5 * i], p[5 * i + 1], p[5 * i + 2], p[5 * i + 3], p[5 * i + 4]
        );
    }
    return qtensor::from_vector(std::move(data), shape);
}

py::array_t<double> qtensor_to_array(const qtensor& x) {
    std::vector<py::ssize_t> shape(x.shape().begin(), x.shape().end());
    shape.push_back(5);
    py::array_t<double> out(shape);
    double* p = out.mutable_data();
    const std::vector<qnumber> data = x.to_vector();
    for (std::size_t i = 0; i < data.size(); ++i) {
        p[5 * i] = data[i].v;
        p[5 * i + 1] = data[i].g1;
        p[5 * i + 2] = data[i].g2;
        p[5 * i + 3] = data[i].x1;
        p[5 * i + 4] = data[i].x2;
    }
    return out;
}

// Network layers on qtensors (activation.h, conv.h, norm.h, pooling.h,
// transform.h).  Weights are numpy arrays in PyTorch's layout, e.g.
// Conv2d(m.weight.detach().double().numpy(), m.bias...).
void export_nn(py::module_& math) {
    using opt = std::optional<darray>;

    // activations (qnumber and qtensor overloads)
    math.def("relu", py::overload_cast<const qnumber&>(&relu), py::arg("x"));
    math.def("relu", py::overload_cast<const qtensor&>(&relu), py::arg("x"));
    math.def(
        "leaky_relu", py::overload_cast<const qnumber&, double>(&leaky_relu),
        py::arg("x"), py::arg("slope") = 0.01
    );
    math.def(
        "leaky_relu", py::overload_cast<const qtensor&, double>(&leaky_relu),
        py::arg("x"), py::arg("slope") = 0.01
    );
    math.def(
        "sigmoid", py::overload_cast<const qnumber&>(&sigmoid), py::arg("x")
    );
    math.def(
        "sigmoid", py::overload_cast<const qtensor&>(&sigmoid), py::arg("x")
    );
    math.def("silu", py::overload_cast<const qnumber&>(&silu), py::arg("x"));
    math.def("silu", py::overload_cast<const qtensor&>(&silu), py::arg("x"));
    math.def("tanh", py::overload_cast<const qnumber&>(&tanh), py::arg("x"));
    math.def("tanh", py::overload_cast<const qtensor&>(&tanh), py::arg("x"));
    math.def(
        "gelu", py::overload_cast<const qnumber&, bool>(&gelu),
        py::arg("x"), py::arg("tanh_approx") = false,
        "torch.nn.GELU (exact erf, or approximate='tanh')"
    );
    math.def(
        "gelu", py::overload_cast<const qtensor&, bool>(&gelu),
        py::arg("x"), py::arg("tanh_approx") = false
    );
    math.def(
        "softplus",
        py::overload_cast<const qnumber&, double, double>(&softplus),
        py::arg("x"), py::arg("beta") = 1.0, py::arg("threshold") = 20.0
    );
    math.def(
        "softplus",
        py::overload_cast<const qtensor&, double, double>(&softplus),
        py::arg("x"), py::arg("beta") = 1.0, py::arg("threshold") = 20.0
    );
    math.def(
        "softmax", &softmax, py::arg("x"), "softmax over the last dimension"
    );
    math.def(
        "add", &add, py::arg("a"), py::arg("b"), py::arg("scale") = 1.0,
        "a + scale * b (residual connection)"
    );
    math.def(
        "mul", &mul, py::arg("x"), py::arg("s"),
        "x * s for a qnumber s (product rule on every element)"
    );

    py::class_<Conv2d>(math, "Conv2d")
        .def(
            py::init([](const darray& weight, const opt& bias,
                        std::size_t stride, std::size_t padding,
                        std::size_t dilation) {
                if (weight.ndim() != 4) {
                    throw std::invalid_argument(
                        "Conv2d weight must be [out, in, kh, kw]"
                    );
                }
                return Conv2d(
                    dim(weight, 1), dim(weight, 0), dim(weight, 2),
                    dim(weight, 3), to_vec(weight), to_vec(bias),
                    stride, padding, dilation
                );
            }),
            py::arg("weight"), py::arg("bias") = py::none(),
            py::arg("stride") = 1, py::arg("padding") = 0,
            py::arg("dilation") = 1,
            "torch.nn.Conv2d on one sample [C_in, H, W] (zero padding)"
        )
        .def("forward", &Conv2d::forward, py::arg("x"))
        .def("__call__", &Conv2d::forward, py::arg("x"))
        .def_readonly("in_channels", &Conv2d::in_channels)
        .def_readonly("out_channels", &Conv2d::out_channels);

    py::class_<Linear>(math, "Linear")
        .def(
            py::init([](const darray& weight, const opt& bias) {
                if (weight.ndim() != 2) {
                    throw std::invalid_argument(
                        "Linear weight must be [out, in]"
                    );
                }
                return Linear(
                    dim(weight, 1), dim(weight, 0), to_vec(weight),
                    to_vec(bias)
                );
            }),
            py::arg("weight"), py::arg("bias") = py::none(),
            "torch.nn.Linear over the last dimension"
        )
        .def("forward", &Linear::forward, py::arg("x"))
        .def("__call__", &Linear::forward, py::arg("x"))
        .def_readonly("in_features", &Linear::in_features)
        .def_readonly("out_features", &Linear::out_features);

    py::class_<LayerNorm>(math, "LayerNorm")
        .def(
            py::init([](std::size_t features, const opt& weight,
                        const opt& bias, double eps) {
                return LayerNorm(features, to_vec(weight), to_vec(bias), eps);
            }),
            py::arg("features"), py::arg("weight") = py::none(),
            py::arg("bias") = py::none(), py::arg("eps") = 1e-5,
            "torch.nn.LayerNorm over the last dimension"
        )
        .def("forward", &LayerNorm::forward, py::arg("x"))
        .def("__call__", &LayerNorm::forward, py::arg("x"));

    py::class_<ChannelLayerNorm2d>(math, "ChannelLayerNorm2d")
        .def(
            py::init([](std::size_t channels, const opt& weight,
                        const opt& bias, double eps) {
                return ChannelLayerNorm2d(
                    channels, to_vec(weight), to_vec(bias), eps
                );
            }),
            py::arg("channels"), py::arg("weight") = py::none(),
            py::arg("bias") = py::none(), py::arg("eps") = 1e-5,
            "layer norm over the channels of [C, H, W] at each pixel"
        )
        .def("forward", &ChannelLayerNorm2d::forward, py::arg("x"))
        .def("__call__", &ChannelLayerNorm2d::forward, py::arg("x"));

    py::class_<BatchNorm2d>(math, "BatchNorm2d")
        .def(
            py::init([](const darray& running_mean, const darray& running_var,
                        const opt& weight, const opt& bias, double eps) {
                return BatchNorm2d(
                    static_cast<std::size_t>(running_mean.size()),
                    to_vec(running_mean), to_vec(running_var),
                    to_vec(weight), to_vec(bias), eps
                );
            }),
            py::arg("running_mean"), py::arg("running_var"),
            py::arg("weight") = py::none(), py::arg("bias") = py::none(),
            py::arg("eps") = 1e-5,
            "torch.nn.BatchNorm2d in eval mode"
        )
        .def("forward", &BatchNorm2d::forward, py::arg("x"))
        .def("__call__", &BatchNorm2d::forward, py::arg("x"));

    // pooling
    math.def(
        "max_pool2d", &max_pool2d, py::arg("x"), py::arg("kernel_size"),
        py::arg("stride") = 0, py::arg("padding") = 0,
        "torch.nn.MaxPool2d (stride 0 = kernel_size)"
    );
    math.def(
        "avg_pool2d", &avg_pool2d, py::arg("x"), py::arg("kernel_size"),
        py::arg("stride") = 0, py::arg("padding") = 0,
        py::arg("count_include_pad") = true,
        "torch.nn.AvgPool2d (stride 0 = kernel_size)"
    );
    math.def(
        "adaptive_avg_pool2d", &adaptive_avg_pool2d, py::arg("x"),
        py::arg("out_h"), py::arg("out_w"), "torch.nn.AdaptiveAvgPool2d"
    );
    math.def(
        "weighted_avg_pool",
        [](const qtensor& x, const darray& weight, bool normalize) {
            return weighted_avg_pool(x, to_vec(weight), normalize);
        },
        py::arg("x"), py::arg("weight"), py::arg("normalize") = true,
        "[C, H, W] -> [C]: sum x w / sum w with a fixed weight map [H, W]"
    );
    math.def(
        "global_avg_pool", &global_avg_pool, py::arg("x"),
        "[C, H, W] -> [C] mean over the pixels"
    );

    // pixel rearrangements
    math.def(
        "rot90", &rot90, py::arg("x"), py::arg("k") = 1,
        "torch.rot90(x, k, dims=(-2, -1))"
    );
    math.def(
        "flip", &flip, py::arg("x"), py::arg("dim"),
        "torch.flip(x, [dim]) for dim 1 (rows) or 2 (columns)"
    );
    math.def(
        "crop2d", &crop2d, py::arg("x"), py::arg("y0"), py::arg("x0"),
        py::arg("h"), py::arg("w"), "x[:, y0:y0 + h, x0:x0 + w]"
    );
    math.def(
        "pad2d", &pad2d, py::arg("x"), py::arg("padding"),
        py::arg("mode") = "constant",
        "F.pad(x, (p, p, p, p), mode) for constant (zero) or replicate"
    );
}

} // namespace

    void
    pyExportMath(py::module_ &m) {
        py::module_ math = m.def_submodule(
            "math",
            "submodule for Math"
        );
        py::class_<qnumber>(math, "qnumber")
            .def(py::init<>())
            .def(
                py::init<double>(),
                "initialize qnumber with double number as value",
                py::arg("v")
            )
            .def(
                py::init<double, double, double, double, double>(),
                "initialize qnumber with double numbers",
                py::arg("v"), py::arg("g1"), py::arg("g2"),
                py::arg("x1"), py::arg("x2")
            )
            .def(
                py::init<std::array<double, 5>&>(),
                "initialize qnumber with a list of double numbers",
                py::arg("data")
            )
            .def_readwrite("v", &qnumber::v)
            .def_readwrite("g1", &qnumber::g1)
            .def_readwrite("g2", &qnumber::g2)
            .def_readwrite("x1", &qnumber::x1)
            .def_readwrite("x2", &qnumber::x2)
            .def(
                "to_array",
                &qnumber::to_array,
                "Return the numpy array in shape (5)"
            )
            .def("__repr__", [](
                const qnumber &tn) {
                return "<qnumber v=" + std::to_string(tn.v) +
                       ", g1=" + std::to_string(tn.g1) +
                       ", g2=" + std::to_string(tn.g2) +
                       ", x1=" + std::to_string(tn.x1) +
                       ", x2=" + std::to_string(tn.x2) + ">";
                }
            )
            .def(-py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self * py::self)
            .def(py::self / py::self)
            .def(py::self + double())
            .def(double() + py::self)
            .def(py::self - double())
            .def(double() - py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self / double())
            .def(double() / py::self);
        math.def("exp", &exp, "Compute the exponential of a qnumber");
        math.def("pow", &pow, "Compute the exponential of a qnumber");

        using size_type = qtensor::size_type;
        py::class_<qtensor>(math, "qtensor")
            .def(py::init<>())
            .def(py::init<const std::vector<size_type>&>())
            .def(py::init<const std::vector<size_type>&, const qnumber&>())
            .def_static(
                "from_flat",
                [](std::vector<qnumber> data,
                   const std::vector<size_type>& shape) {
                    return qtensor::from_vector(std::move(data), shape);
                },
                py::arg("data"),
                py::arg("shape"),
                "Create a qtensor from a flat buffer of qnumbers"
            )
            .def_static(
                "from_image",
                [](std::vector<qnumber> data,
                   size_type height,
                   size_type width) {
                    return qtensor::from_image(
                        std::move(data),
                        height,
                        width
                    );
                },
                py::arg("data"),
                py::arg("height"),
                py::arg("width"),
                "Create a qtensor from ImageQ output"
            )
            .def_property_readonly(
                "shape",
                [](const qtensor& self) { return self.shape(); },
                "Tensor shape expressed as a tuple of dimension lengths"
            )
            .def_property_readonly(
                "strides",
                [](const qtensor& self) { return self.strides(); },
                "Tensor strides in units of qnumbers"
            )
            .def_property_readonly(
                "ndim",
                [](const qtensor& self) { return self.ndim(); },
                "Number of tensor dimensions"
            )
            .def("size", &qtensor::size, "Total number of qnumbers")
            .def("empty", &qtensor::empty, "Return True if the tensor is empty")
            .def(
                "is_contiguous",
                &qtensor::is_contiguous,
                "Return True if the tensor uses contiguous storage"
            )
            .def(
                "reshape",
                &qtensor::reshape,
                py::arg("shape"),
                "Return a reshaped view of the tensor"
            )
            .def(
                "slice",
                &qtensor::slice,
                py::arg("dim"),
                py::arg("start"),
                py::arg("stop"),
                py::arg("step") = 1,
                "Return a view defined by slicing a single dimension"
            )
            .def(
                "select",
                &qtensor::select,
                py::arg("dim"),
                py::arg("index"),
                "Return a view with a dimension fixed to a single index"
            )
            .def(
                "get",
                [](const qtensor& self, const std::vector<size_type>& indices) {
                    return self.at(indices);
                },
                py::arg("indices"),
                "Access a single qnumber using multidimensional indices"
            )
            .def(
                "set",
                [](qtensor& self,
                   const std::vector<size_type>& indices,
                   const qnumber& value) {
                    self.at(indices) = value;
                },
                py::arg("indices"),
                py::arg("value"),
                "Assign a qnumber using multidimensional indices"
            )
            .def(
                "to_list",
                &qtensor::to_vector,
                "Return a flat list of qnumbers in row-major order"
            )
            .def_static(
                "from_array",
                &qtensor_from_array,
                py::arg("array"),
                "qtensor of shape [...] from a float array [..., 5] in the "
                "order (v, g1, g2, x1, x2).  The [5, H, W] output of "
                "ImageQ.prepare_qnumber_image becomes a one-channel "
                "[1, H, W] tensor with np.moveaxis(q, 0, -1)[None]"
            )
            .def(
                "to_array",
                &qtensor_to_array,
                "Return the numpy array [..., 5] (v, g1, g2, x1, x2)"
            );

        py::class_<qmatrix<6, 6>>(math, "qmatrix6")
            .def(py::init<>())
            .def(
                py::init<std::array<std::array<qnumber, 6>, 6>&>(),
                "initialize qmatrix with a 2D list of qnumber",
                py::arg("data")
            )
            .def(
                py::init<std::array<std::array<double, 6>, 6>&>(),
                "initialize qmatrix with a 2D list of double",
                py::arg("data")
            )
            .def(
                py::init<py::array_t<double>&>(),
                "initialize qmatrix with a 2D numpy array",
                py::arg("data")
            )
            .def_readonly("data", &qmatrix<6, 6>::data)
            .def(
                "transpose",
                &qmatrix<6, 6>::transpose,
                "Return the transposed matrix"
            )
            .def(
                "to_array",
                &qmatrix<6, 6>::to_array,
                "Return the numpy array in shape (6, 6, 5)"
            )
            .def(-py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self + qnumber())
            .def(qnumber() + py::self)
            .def(py::self - qnumber())
            .def(qnumber() - py::self)
            .def(py::self * qnumber())
            .def(qnumber() * py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self / double())
            .def(py::self / qnumber())
            .def(py::self * py::self);

        py::class_<qmatrix<8, 8>>(math, "qmatrix8")
            .def(py::init<>())
            .def(
                py::init<std::array<std::array<qnumber, 8>, 8>&>(),
                "initialize qmatrix with a 2D list of qnumber",
                py::arg("data")
            )
            .def(
                py::init<std::array<std::array<double, 8>, 8>&>(),
                "initialize qmatrix with a 2D list of double number",
                py::arg("data")
            )
            .def(
                py::init<py::array_t<double>&>(),
                "initialize qmatrix with a 2D numpy array of double number",
                py::arg("data")
            )
            .def_readonly("data", &qmatrix<8, 8>::data)
            .def(
                "transpose",
                &qmatrix<8, 8>::transpose,
                "Return the transposed matrix"
            )
            .def(
                "to_array",
                &qmatrix<8, 8>::to_array,
                "Return the numpy array in shape (8, 8, 5)"
            )
            .def(-py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self + qnumber())
            .def(qnumber() + py::self)
            .def(py::self - qnumber())
            .def(qnumber() - py::self)
            .def(py::self * qnumber())
            .def(qnumber() * py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self / double())
            .def(py::self / qnumber())
            .def(py::self * py::self);
        math.def("eye6", &eye<6>, "Generate a indentity qmatrix");
        math.def("eye8", &eye<8>, "Generate a indentity qmatrix");

        py::class_<lossNumber> loss_cls(math, "lossNumber");
        loss_cls.def(py::init<>());
#define ANACAL_LOSS_BIND(name) \
        loss_cls.def_readwrite(#name, &lossNumber::name);
        ANACAL_LOSS_FIELDS(ANACAL_LOSS_BIND)
#undef ANACAL_LOSS_BIND
        loss_cls.def_readwrite("n_pix", &lossNumber::n_pix);

        export_nn(math);
    }
} // math
} // anacal
