#include <cstring>

#include "anacal.h"

namespace anacal {
namespace psfmodel {

namespace {

py::tuple stamp_to_py(const Stamp& st) {
    py::array_t<double> arr({st.ny, st.nx});
    std::memcpy(
        arr.mutable_data(), st.data.data(),
        st.data.size() * sizeof(double)
    );
    return py::make_tuple(arr, py::make_tuple(st.x0, st.y0));
}

// The resize_array conventions; delegates to the pure C++
// resize_stamp_to shared with the native ForceTask path.
py::array_t<double> stamp_to_npix(const Stamp& st, int npix) {
    py::array_t<double> out({npix, npix});
    resize_stamp_to(st, npix, out.mutable_data());
    return out;
}

MappingPoly mapping_from_py(
    int order, double cx, double cy, double sx, double sy,
    const py::array_t<double>& cu, const py::array_t<double>& cv
) {
    const int m = order + 1;
    if (cu.ndim() != 2 || cu.shape(0) != m || cu.shape(1) != m ||
        cv.ndim() != 2 || cv.shape(0) != m || cv.shape(1) != m) {
        throw std::runtime_error(
            "psfmodel Error: mapping coefficient arrays must be "
            "(order+1, order+1)"
        );
    }
    MappingPoly mp;
    mp.order = order;
    mp.cx = cx;
    mp.cy = cy;
    mp.sx = sx;
    mp.sy = sy;
    mp.cu.resize(static_cast<std::size_t>(m) * m);
    mp.cv.resize(static_cast<std::size_t>(m) * m);
    auto ru = cu.unchecked<2>();
    auto rv = cv.unchecked<2>();
    for (int a = 0; a < m; ++a) {
        for (int b = 0; b < m; ++b) {
            mp.cu[static_cast<std::size_t>(a) * m + b] = ru(a, b);
            mp.cv[static_cast<std::size_t>(a) * m + b] = rv(a, b);
        }
    }
    return mp;
}

}  // namespace

void
pyExportPsfModel(py::module_& m) {
    py::module_ psfmodel = m.def_submodule(
        "psfmodel",
        "native single-visit PSF models + CoaddPsf warping/combination"
    );

    py::class_<SingleVisitPsf, std::shared_ptr<SingleVisitPsf>>(
        psfmodel, "SingleVisitPsf"
    );

    py::class_<PerSourcePsf, std::shared_ptr<PerSourcePsf>>(
        psfmodel, "PerSourcePsf"
    );

    py::class_<PsfexModel, SingleVisitPsf, std::shared_ptr<PsfexModel>>(
        psfmodel, "PsfexModel"
    )
        .def(py::init([](
                const py::array_t<
                    float, py::array::c_style | py::array::forcecast
                >& comp,
                int w, int h, int nbasis,
                double pixstep,
                int degree,
                const py::array_t<double>& context_offset,
                const py::array_t<double>& context_scale
            ) {
                if (context_offset.size() != 2 ||
                    context_scale.size() != 2) {
                    throw std::runtime_error(
                        "psfmodel Error: context must have 2 dimensions"
                    );
                }
                std::vector<float> cvec(comp.size());
                std::memcpy(
                    cvec.data(), comp.data(),
                    comp.size() * sizeof(float)
                );
                return std::make_shared<PsfexModel>(
                    w, h, nbasis, std::move(cvec), pixstep, degree,
                    std::array<double, 2>{
                        context_offset.at(0), context_offset.at(1)
                    },
                    std::array<double, 2>{
                        context_scale.at(0), context_scale.at(1)
                    }
                );
            }),
            "PSFEx single-visit model from PsfexPsf.getSerializationData "
            "contents: comp is the flat float32 basis-major component "
            "array of length w*h*nbasis.",
            py::arg("comp"),
            py::arg("w"),
            py::arg("h"),
            py::arg("nbasis"),
            py::arg("pixstep"),
            py::arg("degree"),
            py::arg("context_offset"),
            py::arg("context_scale")
        )
        .def("compute_image",
            [](const PsfexModel& self, double x, double y) {
                Stamp st;
                {
                    ScopedGilRelease release;
                    st = self.compute_image(x, y);
                }
                return stamp_to_py(st);
            },
            "PsfexPsf.computeImage: (array, (x0, y0))",
            py::arg("x"), py::arg("y")
        )
        .def("compute_kernel_image",
            [](const PsfexModel& self, double x, double y) {
                Stamp st;
                {
                    ScopedGilRelease release;
                    st = self.compute_kernel_image(x, y);
                }
                return stamp_to_py(st);
            },
            "PsfexPsf.computeKernelImage: (array, (x0, y0))",
            py::arg("x"), py::arg("y")
        );

    py::class_<PiffModel, SingleVisitPsf, std::shared_ptr<PiffModel>>(
        psfmodel, "PiffModel"
    )
        .def(py::init([](
                const py::array_t<
                    double, py::array::c_style | py::array::forcecast
                >& q,
                const py::array_t<
                    int, py::array::c_style | py::array::forcecast
                >& term_i,
                const py::array_t<
                    int, py::array::c_style | py::array::forcecast
                >& term_j,
                int nmodel,
                int stamp,
                double coord_scale,
                int lanczos_n,
                const py::array_t<
                    double, py::array::c_style | py::array::forcecast
                >& dc_coeff
            ) {
                std::vector<double> qv(q.size());
                std::memcpy(
                    qv.data(), q.data(), q.size() * sizeof(double)
                );
                std::vector<int> ti(term_i.size()), tj(term_j.size());
                std::memcpy(
                    ti.data(), term_i.data(),
                    term_i.size() * sizeof(int)
                );
                std::memcpy(
                    tj.data(), term_j.data(),
                    term_j.size() * sizeof(int)
                );
                std::vector<double> dc(dc_coeff.size());
                std::memcpy(
                    dc.data(), dc_coeff.data(),
                    dc_coeff.size() * sizeof(double)
                );
                return std::make_shared<PiffModel>(
                    nmodel, stamp, coord_scale, std::move(qv),
                    std::move(ti), std::move(tj), lanczos_n,
                    std::move(dc)
                );
            }),
            "PIFF (PixelGrid + BasisPolynomial) single-visit model: q is "
            "the (N*N, nterm) interpolation matrix (row = iy*N + ix), "
            "term_i/term_j the u/v exponents per term, coord_scale maps "
            "detector pixels to the interpolation coordinates "
            "(u = coord_scale * x), and dc_coeff the galsim conserve_dc "
            "correction coefficients solved by the loader.",
            py::arg("q"),
            py::arg("term_i"),
            py::arg("term_j"),
            py::arg("nmodel"),
            py::arg("stamp"),
            py::arg("coord_scale"),
            py::arg("lanczos_n"),
            py::arg("dc_coeff")
        )
        .def("compute_image",
            [](const PiffModel& self, double x, double y) {
                Stamp st;
                {
                    ScopedGilRelease release;
                    st = self.compute_image(x, y);
                }
                return stamp_to_py(st);
            },
            "PiffPsf.computeImage: (array, (x0, y0))",
            py::arg("x"), py::arg("y")
        )
        .def("compute_kernel_image",
            [](const PiffModel& self, double x, double y) {
                Stamp st;
                {
                    ScopedGilRelease release;
                    st = self.compute_kernel_image(x, y);
                }
                return stamp_to_py(st);
            },
            "PiffPsf.computeKernelImage: (array, (x0, y0))",
            py::arg("x"), py::arg("y")
        );

    py::class_<GridPsfModel, PerSourcePsf, std::shared_ptr<GridPsfModel>>(
        psfmodel, "GridPsfModel"
    )
        .def(py::init([](
                const py::array_t<
                    double, py::array::c_style | py::array::forcecast
                >& stamps,
                double dx,
                double dy
            ) {
                if (stamps.ndim() != 4 ||
                    stamps.shape(2) != stamps.shape(3)) {
                    throw std::runtime_error(
                        "psfmodel Error: stamps must be "
                        "(ny, nx, npix, npix)"
                    );
                }
                std::vector<double> sv(stamps.size());
                std::memcpy(
                    sv.data(), stamps.data(),
                    stamps.size() * sizeof(double)
                );
                return std::make_shared<GridPsfModel>(
                    static_cast<int>(stamps.shape(0)),
                    static_cast<int>(stamps.shape(1)),
                    static_cast<int>(stamps.shape(2)),
                    dx, dy, std::move(sv)
                );
            }),
            "Per-source PSF from a coarse stamp grid (nearest cell, no "
            "interpolation): cell (i, j) covers x in [j*dx, (j+1)*dx), "
            "y in [i*dy, (i+1)*dy); outside positions clamp to the edge.",
            py::arg("stamps"),
            py::arg("dx"),
            py::arg("dy")
        )
        .def("draw",
            [](const GridPsfModel& self, double x, double y, int npix) {
                Stamp st;
                {
                    ScopedGilRelease release;
                    st = self.compute_image(x, y);
                }
                return stamp_to_npix(st, npix);
            },
            "Cell stamp at (x, y), centre cropped/padded to (npix, npix).",
            py::arg("x"), py::arg("y"), py::arg("npix")
        );

    py::class_<CoaddPsfModel, PerSourcePsf, std::shared_ptr<CoaddPsfModel>>(
        psfmodel, "CoaddPsfModel"
    )
        .def(py::init<int, int>(),
            "CoaddPsf evaluator; lanczos_order/cache_size must come from "
            "the persisted CoaddPsf record (warpingkernelname, cachesize).",
            py::arg("lanczos_order"),
            py::arg("cache_size")
        )
        .def("add_component",
            [](CoaddPsfModel& self,
               const std::shared_ptr<SingleVisitPsf>& model,
               double weight,
               double bx0, double by0, double bx1, double by1,
               const std::optional<py::array_t<double>>& polygon,
               int map_order,
               double map_cx, double map_cy,
               double map_sx, double map_sy,
               const py::array_t<double>& map_cu,
               const py::array_t<double>& map_cv
            ) {
                CoaddComponent c;
                c.model = model;
                c.weight = weight;
                c.bx0 = bx0;
                c.by0 = by0;
                c.bx1 = bx1;
                c.by1 = by1;
                if (polygon.has_value()) {
                    if (polygon->ndim() != 2 || polygon->shape(1) != 2) {
                        throw std::runtime_error(
                            "psfmodel Error: polygon must be (nvertex, 2)"
                        );
                    }
                    auto r = polygon->unchecked<2>();
                    for (ssize_t i = 0; i < polygon->shape(0); ++i) {
                        c.poly_x.push_back(r(i, 0));
                        c.poly_y.push_back(r(i, 1));
                    }
                }
                c.map = mapping_from_py(
                    map_order, map_cx, map_cy, map_sx, map_sy,
                    map_cu, map_cv
                );
                self.components.push_back(std::move(c));
            },
            "Add one coadd input: its single-visit model, weight, the "
            "visit bbox as the OUTER float box (min-0.5 .. max+0.5), the "
            "validPolygon vertices in the visit frame (or None), and the "
            "fitted coadd->visit polynomial mapping.",
            py::arg("model"),
            py::arg("weight"),
            py::arg("bx0"), py::arg("by0"),
            py::arg("bx1"), py::arg("by1"),
            py::arg("polygon"),
            py::arg("map_order"),
            py::arg("map_cx"), py::arg("map_cy"),
            py::arg("map_sx"), py::arg("map_sy"),
            py::arg("map_cu"), py::arg("map_cv")
        )
        .def_property_readonly("n_components",
            [](const CoaddPsfModel& self) {
                return self.components.size();
            }
        )
        .def("contains", &CoaddPsfModel::contains,
            "True when at least one coadd input covers the point (the "
            "exact rule evaluation uses; False predicts the DM "
            "InvalidPsfError).",
            py::arg("x"), py::arg("y")
        )
        .def("compute_kernel_image",
            [](const CoaddPsfModel& self, double x, double y) {
                Stamp st;
                {
                    ScopedGilRelease release;
                    st = self.compute_kernel_image(x, y);
                }
                return stamp_to_py(st);
            },
            "CoaddPsf.computeKernelImage: (array, (x0, y0))",
            py::arg("x"), py::arg("y")
        )
        .def("compute_image",
            [](const CoaddPsfModel& self, double x, double y) {
                Stamp st;
                {
                    ScopedGilRelease release;
                    st = self.compute_image(x, y);
                }
                return stamp_to_py(st);
            },
            "CoaddPsf.computeImage: (array, (x0, y0))",
            py::arg("x"), py::arg("y")
        )
        .def("draw",
            [](const CoaddPsfModel& self, double x, double y, int npix) {
                Stamp st;
                {
                    ScopedGilRelease release;
                    st = self.compute_image(x, y);
                }
                return stamp_to_npix(st, npix);
            },
            "computeImage cropped/padded to (npix, npix) with the xlens "
            "resize_array convention -- the LsstPsf.draw contract.",
            py::arg("x"), py::arg("y"), py::arg("npix")
        );

    psfmodel.def(
        "resize_array",
        [](
            const py::array_t<
                double, py::array::c_style | py::array::forcecast
            >& arr,
            std::pair<int, int> target_shape
        ) {
            const int ny = target_shape.first;
            const int nx = target_shape.second;
            if (arr.ndim() != 2) {
                throw std::runtime_error(
                    "psfmodel Error: resize_array input must be 2-D"
                );
            }
            Stamp st;
            st.ny = static_cast<int>(arr.shape(0));
            st.nx = static_cast<int>(arr.shape(1));
            st.data.assign(
                arr.data(),
                arr.data() + static_cast<std::size_t>(st.ny) * st.nx
            );
            py::array_t<double> out({ny, nx});
            resize_stamp_to(st, ny, nx, out.mutable_data());
            return out;
        },
        "Centre crop / zero-pad a 2-D array to target_shape = "
        "(height, width).  The single implementation of this "
        "convention: crop start (in-out)//2, rows pad bottom-heavy, "
        "columns pad top-heavy.  Any dtype is accepted and the result "
        "is float64.",
        py::arg("array"), py::arg("target_shape") = std::pair<int, int>(64, 64)
    );
}

}  // namespace psfmodel
}  // namespace anacal
