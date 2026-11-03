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

// The four tensor types: scalar float / double, 2 or 4 tangents.  The
// Python names say the scalar and the number of tangents; the plain
// ``qtensor`` is the Task's (double, 4 tangents).
template <typename T, int NT>
struct tensor_name;
template <> struct tensor_name<double, 4> {
    static constexpr const char* tensor = "qtensor";
    static constexpr const char* number = "qnumber";
};
template <> struct tensor_name<double, 2> {
    static constexpr const char* tensor = "qtensor2";
    static constexpr const char* number = "qnumber2";
};
template <> struct tensor_name<float, 4> {
    static constexpr const char* tensor = "qtensorf";
    static constexpr const char* number = "qnumberf";
};
template <> struct tensor_name<float, 2> {
    static constexpr const char* tensor = "qtensor2f";
    static constexpr const char* number = "qnumber2f";
};

template <typename T>
const char* dtype_name() {
    return std::is_same<T, float>::value ? "float32" : "float64";
}

// [..., ncomp] numpy array <-> qtensor of shape [...]
template <typename T, int NT>
qtensor_t<T, NT> qtensor_from_array(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& a
) {
    constexpr int nc = qtensor_t<T, NT>::ncomp;
    if (a.ndim() < 2 || a.shape(a.ndim() - 1) != nc) {
        throw std::invalid_argument(
            std::string(tensor_name<T, NT>::tensor)
            + ".from_array expects an array [..., " + std::to_string(nc) + "]"
        );
    }
    std::vector<std::size_t> shape;
    for (py::ssize_t i = 0; i + 1 < a.ndim(); ++i) {
        shape.push_back(static_cast<std::size_t>(a.shape(i)));
    }
    qtensor_t<T, NT> out(shape);
    const std::size_t n = static_cast<std::size_t>(a.size() / nc);
    const T* p = a.data();
    for (int c = 0; c < nc; ++c) {
        T* plane = out.plane(c);
        for (std::size_t i = 0; i < n; ++i) {
            plane[i] = p[nc * i + c];
        }
    }
    return out;
}

template <typename T, int NT>
py::array_t<T> qtensor_to_array(const qtensor_t<T, NT>& x) {
    constexpr int nc = qtensor_t<T, NT>::ncomp;
    const qtensor_t<T, NT> xc = contiguous(x);
    std::vector<py::ssize_t> shape(xc.shape().begin(), xc.shape().end());
    shape.push_back(nc);
    py::array_t<T> out(shape);
    T* p = out.mutable_data();
    const std::size_t n = xc.size();
    for (int c = 0; c < nc; ++c) {
        const T* plane = xc.plane(c);
        for (std::size_t i = 0; i < n; ++i) {
            p[nc * i + c] = plane[i];
        }
    }
    return out;
}

// x converted to the tensor type named by (dtype, tangents), as a Python
// object (the four types are distinct Python classes)
template <typename T, int NT>
py::object qtensor_astype(
    const qtensor_t<T, NT>& x, const std::string& dtype, int tangents
) {
    const bool f32 = (dtype == "float32" || dtype == "float");
    const bool f64 = (dtype == "float64" || dtype == "double");
    if (!f32 && !f64) {
        throw std::invalid_argument("astype: dtype must be float32 or float64");
    }
    if (tangents != 2 && tangents != 4) {
        throw std::invalid_argument("astype: tangents must be 2 or 4");
    }
    if (f64 && tangents == 4) return py::cast(x.template convert<double, 4>());
    if (f64) return py::cast(x.template convert<double, 2>());
    if (tangents == 4) return py::cast(x.template convert<float, 4>());
    return py::cast(x.template convert<float, 2>());
}

// The non-default qnumber types get the arithmetic and the array access;
// the default ``qnumber`` keeps its full historical binding below.
template <typename T, int NT>
void bind_qnumber_small(py::module_& math) {
    using Q = qnumber_t<T, NT>;
    constexpr int nc = Q::ncomp;
    auto cls = py::class_<Q>(math, tensor_name<T, NT>::number)
        .def(py::init<>())
        .def(py::init<T>(), py::arg("v"))
        .def(py::init<T, T, T>(), py::arg("v"), py::arg("g1"), py::arg("g2"))
        .def(
            py::init([](const std::array<T, nc>& data) { return Q(data); }),
            py::arg("data")
        )
        .def_readwrite("v", &Q::v)
        .def_readwrite("g1", &Q::g1)
        .def_readwrite("g2", &Q::g2)
        .def("to_array", &Q::to_array, "the components (v, g1, g2[, x1, x2])")
        .def("__repr__", [](const Q& q) {
            std::ostringstream ss;
            ss << "<" << tensor_name<T, NT>::number << " " << q << ">";
            return ss.str();
        })
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
    if constexpr (NT == 4) {
        cls.def_readwrite("x1", &Q::x1).def_readwrite("x2", &Q::x2);
        cls.def(
            py::init<T, T, T, T, T>(), py::arg("v"), py::arg("g1"),
            py::arg("g2"), py::arg("x1"), py::arg("x2")
        );
    }
}

template <typename T, int NT>
void bind_qtensor(py::module_& math) {
    using Tensor = qtensor_t<T, NT>;
    using Q = qnumber_t<T, NT>;
    using size_type = typename Tensor::size_type;
    using tarray = py::array_t<T, py::array::c_style | py::array::forcecast>;
    constexpr int nc = Tensor::ncomp;
    const std::string doc_arr = std::string("[..., ") + std::to_string(nc)
        + "] array of " + dtype_name<T>() + " in the order (v, g1, g2"
        + (NT == 4 ? ", x1, x2)" : ")");
    py::class_<Tensor>(math, tensor_name<T, NT>::tensor)
        .def(py::init<>())
        .def(py::init<const std::vector<size_type>&>())
        .def(py::init<const std::vector<size_type>&, const Q&>())
        .def_static(
            "from_flat",
            [](std::vector<Q> data, const std::vector<size_type>& shape) {
                return Tensor::from_vector(std::move(data), shape);
            },
            py::arg("data"), py::arg("shape"),
            "Create a tensor from a flat buffer of qnumbers"
        )
        .def_static(
            "from_image",
            [](std::vector<Q> data, size_type height, size_type width) {
                return Tensor::from_image(std::move(data), height, width);
            },
            py::arg("data"), py::arg("height"), py::arg("width"),
            "Create a tensor from ImageQ output"
        )
        .def_property_readonly(
            "shape", [](const Tensor& self) { return self.shape(); },
            "Tensor shape expressed as a tuple of dimension lengths"
        )
        .def_property_readonly(
            "strides", [](const Tensor& self) { return self.strides(); },
            "Tensor strides in units of qnumbers"
        )
        .def_property_readonly(
            "ndim", [](const Tensor& self) { return self.ndim(); },
            "Number of tensor dimensions"
        )
        .def_property_readonly_static(
            "dtype", [](py::object) { return dtype_name<T>(); }
        )
        .def_property_readonly_static(
            "tangents", [](py::object) { return NT; }
        )
        .def("size", &Tensor::size, "Total number of qnumbers")
        .def("empty", &Tensor::empty, "Return True if the tensor is empty")
        .def(
            "is_contiguous", &Tensor::is_contiguous,
            "Return True if the tensor uses contiguous storage"
        )
        .def(
            "reshape", &Tensor::reshape, py::arg("shape"),
            "Return a reshaped view of the tensor"
        )
        .def(
            "slice", &Tensor::slice, py::arg("dim"), py::arg("start"),
            py::arg("stop"), py::arg("step") = 1,
            "Return a view defined by slicing a single dimension"
        )
        .def(
            "select", &Tensor::select, py::arg("dim"), py::arg("index"),
            "Return a view with a dimension fixed to a single index"
        )
        .def(
            "get",
            [](const Tensor& self, const std::vector<size_type>& indices) {
                return self.at(indices);
            },
            py::arg("indices"),
            "Access a single qnumber using multidimensional indices"
        )
        .def(
            "set",
            [](Tensor& self, const std::vector<size_type>& indices, const Q& value) {
                self.set(indices, value);
            },
            py::arg("indices"), py::arg("value"),
            "Assign a qnumber using multidimensional indices"
        )
        .def(
            "to_list", &Tensor::to_vector,
            "Return a flat list of qnumbers in row-major order"
        )
        .def_static(
            "from_array",
            [](const tarray& a) { return qtensor_from_array<T, NT>(a); },
            py::arg("array"),
            ("tensor of shape [...] from a " + doc_arr
             + ".  The [5, H, W] output of ImageQ.prepare_qnumber_image "
               "becomes a one-channel [1, H, W] tensor with "
               "np.moveaxis(q, 0, -1)[None]").c_str()
        )
        .def(
            "to_array", &qtensor_to_array<T, NT>,
            ("Return the numpy " + doc_arr).c_str()
        )
        .def(
            "astype", &qtensor_astype<T, NT>,
            py::arg("dtype") = "float32", py::arg("tangents") = 2,
            "The same elements as another tensor type: dtype float32 / "
            "float64, 2 tangents (v, g1, g2: a stamp measured about a fixed "
            "centre) or 4 (with the position tangents x1, x2: a cell).  "
            "Dropped tangents are lost, added ones are zero."
        );
}

// Network layers on qtensors (activation.h, conv.h, norm.h, pooling.h,
// transform.h) for one tensor type.  Weights are numpy arrays in
// PyTorch's layout, e.g. Conv2d(m.weight.detach().double().numpy(), ...).
template <typename T, int NT>
void bind_nn_functions(py::module_& math) {
    using Tensor = qtensor_t<T, NT>;
    using Q = qnumber_t<T, NT>;
    math.def("relu", py::overload_cast<const Q&>(&relu<T, NT>), py::arg("x"));
    math.def("relu", py::overload_cast<const Tensor&>(&relu<T, NT>), py::arg("x"));
    math.def(
        "leaky_relu", py::overload_cast<const Q&, double>(&leaky_relu<T, NT>),
        py::arg("x"), py::arg("slope") = 0.01
    );
    math.def(
        "leaky_relu", py::overload_cast<const Tensor&, double>(&leaky_relu<T, NT>),
        py::arg("x"), py::arg("slope") = 0.01
    );
    math.def("sigmoid", py::overload_cast<const Q&>(&sigmoid<T, NT>), py::arg("x"));
    math.def("sigmoid", py::overload_cast<const Tensor&>(&sigmoid<T, NT>), py::arg("x"));
    math.def("silu", py::overload_cast<const Q&>(&silu<T, NT>), py::arg("x"));
    math.def("silu", py::overload_cast<const Tensor&>(&silu<T, NT>), py::arg("x"));
    math.def("tanh", py::overload_cast<const Q&>(&tanh<T, NT>), py::arg("x"));
    math.def("tanh", py::overload_cast<const Tensor&>(&tanh<T, NT>), py::arg("x"));
    math.def(
        "gelu", py::overload_cast<const Q&, bool>(&gelu<T, NT>),
        py::arg("x"), py::arg("tanh_approx") = false,
        "torch.nn.GELU (exact erf, or approximate='tanh')"
    );
    math.def(
        "gelu", py::overload_cast<const Tensor&, bool>(&gelu<T, NT>),
        py::arg("x"), py::arg("tanh_approx") = false
    );
    math.def(
        "softplus", py::overload_cast<const Q&, double, double>(&softplus<T, NT>),
        py::arg("x"), py::arg("beta") = 1.0, py::arg("threshold") = 20.0
    );
    math.def(
        "softplus",
        py::overload_cast<const Tensor&, double, double>(&softplus<T, NT>),
        py::arg("x"), py::arg("beta") = 1.0, py::arg("threshold") = 20.0
    );
    math.def(
        "softmax", &softmax<T, NT>, py::arg("x"),
        "softmax over the last dimension"
    );
    math.def(
        "add", &add<T, NT>, py::arg("a"), py::arg("b"), py::arg("scale") = 1.0,
        "a + scale * b (residual connection)"
    );
    math.def(
        "mul", &mul<T, NT>, py::arg("x"), py::arg("s"),
        "x * s for a qnumber s (product rule on every element)"
    );

    // pooling
    math.def(
        "max_pool2d", &max_pool2d<T, NT>, py::arg("x"), py::arg("kernel_size"),
        py::arg("stride") = 0, py::arg("padding") = 0,
        "torch.nn.MaxPool2d (stride 0 = kernel_size)"
    );
    math.def(
        "avg_pool2d", &avg_pool2d<T, NT>, py::arg("x"), py::arg("kernel_size"),
        py::arg("stride") = 0, py::arg("padding") = 0,
        py::arg("count_include_pad") = true,
        "torch.nn.AvgPool2d (stride 0 = kernel_size)"
    );
    math.def(
        "adaptive_avg_pool2d", &adaptive_avg_pool2d<T, NT>, py::arg("x"),
        py::arg("out_h"), py::arg("out_w"), "torch.nn.AdaptiveAvgPool2d"
    );
    math.def(
        "weighted_avg_pool",
        [](const Tensor& x, const darray& weight, bool normalize) {
            return weighted_avg_pool(x, to_vec(weight), normalize);
        },
        py::arg("x"), py::arg("weight"), py::arg("normalize") = true,
        "[C, H, W] -> [C]: sum x w / sum w with a fixed weight map [H, W]"
    );
    math.def(
        "global_avg_pool", &global_avg_pool<T, NT>, py::arg("x"),
        "[C, H, W] -> [C] mean over the pixels"
    );

    // pixel rearrangements
    math.def(
        "rot90", &rot90<T, NT>, py::arg("x"), py::arg("k") = 1,
        "torch.rot90(x, k, dims=(-2, -1))"
    );
    math.def(
        "flip", &flip<T, NT>, py::arg("x"), py::arg("dim"),
        "torch.flip(x, [dim]) for dim 1 (rows) or 2 (columns)"
    );
    math.def(
        "crop2d", &crop2d<T, NT>, py::arg("x"), py::arg("y0"), py::arg("x0"),
        py::arg("h"), py::arg("w"), "x[:, y0:y0 + h, x0:x0 + w]"
    );
    math.def(
        "pad2d", &pad2d<T, NT>, py::arg("x"), py::arg("padding"),
        py::arg("mode") = "constant",
        "F.pad(x, (p, p, p, p), mode) for constant (zero) or replicate"
    );
}

// forward / __call__ of a layer for one tensor type
template <typename Layer, typename T, int NT>
void bind_forward(py::class_<Layer>& cls) {
    cls.def(
        "forward", &Layer::template forward<T, NT>, py::arg("x")
    );
    cls.def(
        "__call__", &Layer::template forward<T, NT>, py::arg("x")
    );
}

template <typename Layer>
void bind_forwards(py::class_<Layer>& cls) {
    bind_forward<Layer, double, 4>(cls);
    bind_forward<Layer, double, 2>(cls);
    bind_forward<Layer, float, 4>(cls);
    bind_forward<Layer, float, 2>(cls);
}


// The fused D4 shape-network runner, one class per scalar type; measure()
// accepts the 2- and the 4-tangent stamp of that scalar.
template <typename T>
void bind_d4net(py::module_& math, const char* name) {
    using Net = D4ShapeNet<T>;
    using Block = typename Net::Block;
    auto tovec = [](const darray& a) {
        return std::vector<T>(a.data(), a.data() + a.size());
    };
    py::class_<Net>(math, name)
        .def(
            py::init([tovec](const std::vector<py::tuple>& blocks,
                             const std::vector<darray>& head_e1,
                             const std::vector<darray>& head_e2,
                             const darray& wnorm, const darray& wpool,
                             double res, double eps) {
                std::vector<Block> blks;
                for (const py::tuple& t : blocks) {
                    if (t.size() != 5) {
                        throw std::invalid_argument(
                            "block = (conv_w [co, ci, 3, 3], conv_b, norm_w, norm_b, skip_w or None)"
                        );
                    }
                    const darray w = t[0].cast<darray>();
                    if (w.ndim() != 4 || w.shape(2) != 3 || w.shape(3) != 3) {
                        throw std::invalid_argument("conv_w must be [co, ci, 3, 3]");
                    }
                    Block b;
                    b.cout = dim(w, 0);
                    b.cin = dim(w, 1);
                    b.w = tovec(w);
                    b.b = tovec(t[1].cast<darray>());
                    b.nw = tovec(t[2].cast<darray>());
                    b.nb = tovec(t[3].cast<darray>());
                    if (!t[4].is_none()) {
                        b.skip = tovec(t[4].cast<darray>());
                    }
                    blks.push_back(std::move(b));
                }
                if (head_e1.size() != 4 || head_e2.size() != 4) {
                    throw std::invalid_argument("heads: 4 weight matrices [out, in] each");
                }
                std::vector<std::size_t> dims;
                dims.push_back(dim(head_e1[0], 1));
                for (const darray& h : head_e1) dims.push_back(dim(h, 0));
                std::vector<std::vector<T>> h1, h2;
                for (const darray& h : head_e1) h1.push_back(tovec(h));
                for (const darray& h : head_e2) h2.push_back(tovec(h));
                if (wnorm.ndim() != 2 || wnorm.shape(0) != wnorm.shape(1)) {
                    throw std::invalid_argument("wnorm must be [n, n]");
                }
                const std::size_t n = dim(wnorm, 0);
                return Net(
                    std::move(blks), std::move(h1), std::move(h2), std::move(dims),
                    tovec(wnorm), tovec(wpool), n, static_cast<T>(res), static_cast<T>(eps)
                );
            }),
            py::arg("blocks"), py::arg("head_e1"), py::arg("head_e2"),
            py::arg("wnorm"), py::arg("wpool"), py::arg("res") = 0.1,
            py::arg("eps") = 1e-5,
            "Forward8_fixW_CNN from its weights: blocks = [(conv_w, conv_b, "
            "norm_w, norm_b, skip_w | None), ...], the two heads as 4 linear "
            "weights each, the brightness and pooling weight maps [n, n]"
        )
        .def_property_readonly("uses_avx2", &Net::uses_avx2)
        .def("set_profile", &Net::set_profile, py::arg("on"))
        .def("profile", [](const Net& net) {
            static const char* names[9] = {
                "orbit", "conv_direct", "wino_prepare", "wino_gemm",
                "wino_output", "skip", "epilogue", "accumulate", "heads"
            };
            py::dict d;
            const auto p = net.profile();
            for (int i = 0; i < 9; ++i) d[names[i]] = p[i];
            return d;
        }, "accumulated stage times in seconds (after set_profile(True))")
        .def(
            "measure",
            [](Net& net, const qtensor_t<T, 2>& stamp) {
                const auto r = net.template measure<2>(stamp);
                py::array_t<T> out({2, 3});
                std::copy(r.begin(), r.end(), out.mutable_data());
                return out;
            },
            py::arg("stamp"),
            "(e1, e2) x (v, g1, g2) of a [1, n, n] stamp with two tangents"
        )
        .def(
            "measure",
            [](Net& net, const qtensor_t<T, 4>& stamp) {
                const auto r = net.template measure<4>(stamp);
                py::array_t<T> out({2, 5});
                std::copy(r.begin(), r.end(), out.mutable_data());
                return out;
            },
            py::arg("stamp"),
            "(e1, e2) x (v, g1, g2, x1, x2) of a [1, n, n] stamp with four tangents"
        );
}

void export_nn(py::module_& math) {
    using opt = std::optional<darray>;

    bind_nn_functions<double, 4>(math);
    bind_nn_functions<double, 2>(math);
    bind_nn_functions<float, 4>(math);
    bind_nn_functions<float, 2>(math);

    py::class_<Conv2d> conv(math, "Conv2d");
    conv.def(
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
        "torch.nn.Conv2d on one sample [C_in, H, W] (zero padding); "
        "stride 1 / dilation 1 run the blocked SIMD kernel"
    );
    conv.def_readonly("in_channels", &Conv2d::in_channels);
    conv.def_readonly("out_channels", &Conv2d::out_channels);
    bind_forwards(conv);

    py::class_<Linear> linear(math, "Linear");
    linear.def(
        py::init([](const darray& weight, const opt& bias) {
            if (weight.ndim() != 2) {
                throw std::invalid_argument("Linear weight must be [out, in]");
            }
            return Linear(
                dim(weight, 1), dim(weight, 0), to_vec(weight), to_vec(bias)
            );
        }),
        py::arg("weight"), py::arg("bias") = py::none(),
        "torch.nn.Linear over the last dimension"
    );
    linear.def_readonly("in_features", &Linear::in_features);
    linear.def_readonly("out_features", &Linear::out_features);
    bind_forwards(linear);

    py::class_<LayerNorm> lnorm(math, "LayerNorm");
    lnorm.def(
        py::init([](std::size_t features, const opt& weight,
                    const opt& bias, double eps) {
            return LayerNorm(features, to_vec(weight), to_vec(bias), eps);
        }),
        py::arg("features"), py::arg("weight") = py::none(),
        py::arg("bias") = py::none(), py::arg("eps") = 1e-5,
        "torch.nn.LayerNorm over the last dimension"
    );
    bind_forwards(lnorm);

    py::class_<ChannelLayerNorm2d> cnorm(math, "ChannelLayerNorm2d");
    cnorm.def(
        py::init([](std::size_t channels, const opt& weight,
                    const opt& bias, double eps) {
            return ChannelLayerNorm2d(
                channels, to_vec(weight), to_vec(bias), eps
            );
        }),
        py::arg("channels"), py::arg("weight") = py::none(),
        py::arg("bias") = py::none(), py::arg("eps") = 1e-5,
        "layer norm over the channels of [C, H, W] at each pixel"
    );
    bind_forwards(cnorm);

    py::class_<BatchNorm2d> bnorm(math, "BatchNorm2d");
    bnorm.def(
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
    );
    bind_forwards(bnorm);

    bind_d4net<float>(math, "D4ShapeNet32");
    bind_d4net<double>(math, "D4ShapeNet64");

    math.def(
        "cpu_has_avx2_fma", &conv_detail::cpu_has_avx2_fma,
        "whether the convolution runs its AVX2 + FMA kernel on this CPU"
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
        math.def("exp", &exp<double, 4>, "Compute the exponential of a qnumber");
        math.def("pow", &pow<double, 4>, "Compute the exponential of a qnumber");

        bind_qnumber_small<double, 2>(math);
        bind_qnumber_small<float, 4>(math);
        bind_qnumber_small<float, 2>(math);

        bind_qtensor<double, 4>(math);
        bind_qtensor<double, 2>(math);
        bind_qtensor<float, 4>(math);
        bind_qtensor<float, 2>(math);

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
