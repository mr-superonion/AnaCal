#ifndef ANACAL_MATH_TENSOR_H
#define ANACAL_MATH_TENSOR_H

#include "anacal/stdafx.h"
#include "anacal/math/qnumber.h"

#include <cstddef>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

// Tensor of qnumber_t<T, NT>, stored as a structure of arrays: one plane
// of T per component (v, g1, g2[, x1, x2]), all planes sharing the shape,
// the strides and the offset.  Every network layer is then a loop over
// contiguous planes (which vectorises), and the convolution kernel reads
// the planes directly.  Views (reshape, slice, select) share the storage.

namespace anacal {
namespace math {

template <typename T, int NT>
class qtensor_t {
public:
    using size_type = std::size_t;
    using value_type = T;
    using qnumber_type = qnumber_t<T, NT>;
    static constexpr int ntangent = NT;
    static constexpr int ncomp = NT + 1;

    qtensor_t()
        : data_(std::make_shared<std::vector<T>>()), plane_(0) {}

    explicit qtensor_t(const std::vector<size_type>& shape)
        : data_(std::make_shared<std::vector<T>>(
              static_cast<size_type>(ncomp) * compute_size(shape), T(0)
          )),
          plane_(compute_size(shape)),
          shape_(shape),
          strides_(compute_contiguous_strides(shape)),
          offset_(0) {}

    qtensor_t(
        const std::vector<size_type>& shape,
        const qnumber_type& fill_value
    )
        : qtensor_t(shape) {
        for (int c = 0; c < ncomp; ++c) {
            std::fill(
                data_->begin() + c * plane_, data_->begin() + (c + 1) * plane_,
                fill_value.comp(c)
            );
        }
    }

    qtensor_t(const qtensor_t&) = default;
    qtensor_t(qtensor_t&&) noexcept = default;
    qtensor_t& operator=(const qtensor_t&) = default;
    qtensor_t& operator=(qtensor_t&&) noexcept = default;
    ~qtensor_t() = default;

    static qtensor_t from_shape(const std::vector<size_type>& shape) {
        return qtensor_t(shape);
    }

    static qtensor_t from_shape(
        const std::vector<size_type>& shape,
        const qnumber_type& fill_value
    ) {
        return qtensor_t(shape, fill_value);
    }

    static qtensor_t from_vector(
        const std::vector<qnumber_type>& data,
        const std::vector<size_type>& shape
    ) {
        const size_type expected = compute_size(shape);
        if (data.size() != expected) {
            std::ostringstream ss;
            ss << "qtensor::from_vector expects " << expected
               << " elements but received " << data.size();
            throw std::invalid_argument(ss.str());
        }
        qtensor_t out(shape);
        for (int c = 0; c < ncomp; ++c) {
            T* p = out.plane(c);
            for (size_type i = 0; i < expected; ++i) {
                p[i] = data[i].comp(c);
            }
        }
        return out;
    }

    static qtensor_t from_vector(
        std::vector<qnumber_type>&& data,
        const std::vector<size_type>& shape
    ) {
        return from_vector(static_cast<const std::vector<qnumber_type>&>(data), shape);
    }

    static qtensor_t from_image(
        std::vector<qnumber_type>&& data,
        size_type height,
        size_type width
    ) {
        return from_vector(data, {height, width});
    }

    size_type ndim() const noexcept { return shape_.size(); }

    const std::vector<size_type>& shape() const noexcept { return shape_; }

    const std::vector<size_type>& strides() const noexcept { return strides_; }

    size_type size() const noexcept { return compute_size(shape_); }

    bool empty() const noexcept { return size() == 0; }

    // One element, gathered from the planes.
    qnumber_type at(const std::vector<size_type>& indices) const {
        const size_type index = linear_index(indices);
        qnumber_type q;
        for (int c = 0; c < ncomp; ++c) {
            q.comp(c) = (*data_)[c * plane_ + index];
        }
        return q;
    }

    void set(const std::vector<size_type>& indices, const qnumber_type& value) {
        const size_type index = linear_index(indices);
        for (int c = 0; c < ncomp; ++c) {
            (*data_)[c * plane_ + index] = value.comp(c);
        }
    }

    // Plane c (component order v, g1, g2, x1, x2) of a contiguous tensor.
    T* plane(int c) {
        if (!is_contiguous()) {
            throw std::runtime_error("plane() requires contiguous storage");
        }
        return data_->data() + c * plane_ + offset_;
    }

    const T* plane(int c) const {
        if (!is_contiguous()) {
            throw std::runtime_error("plane() requires contiguous storage");
        }
        return data_->data() + c * plane_ + offset_;
    }

    bool is_contiguous() const noexcept {
        size_type expected_stride = 1;
        for (std::ptrdiff_t dim = static_cast<std::ptrdiff_t>(shape_.size()) - 1;
             dim >= 0; --dim) {
            if (shape_[static_cast<size_type>(dim)] == 0) {
                return true;
            }
            if (strides_[static_cast<size_type>(dim)] != expected_stride) {
                return false;
            }
            expected_stride *= shape_[static_cast<size_type>(dim)];
        }
        return true;
    }

    qtensor_t reshape(const std::vector<size_type>& new_shape) const {
        if (!is_contiguous()) {
            throw std::runtime_error("reshape requires contiguous storage");
        }
        if (compute_size(new_shape) != size()) {
            throw std::invalid_argument("reshape size mismatch");
        }
        return qtensor_t(
            data_, plane_, new_shape, compute_contiguous_strides(new_shape), offset_
        );
    }

    qtensor_t slice(
        size_type dim,
        size_type start,
        size_type stop,
        size_type step = 1
    ) const {
        if (dim >= shape_.size()) {
            throw std::out_of_range("slice dimension out of bounds");
        }
        ensure_step_valid(step);
        if (start > stop || stop > shape_[dim]) {
            throw std::out_of_range("invalid slice range");
        }
        size_type length = stop - start;
        size_type new_extent = length == 0 ? 0 : (length + step - 1) / step;
        std::vector<size_type> new_shape = shape_;
        std::vector<size_type> new_strides = strides_;
        new_shape[dim] = new_extent;
        new_strides[dim] *= step;
        size_type new_offset = offset_ + start * strides_[dim];
        return qtensor_t(data_, plane_, new_shape, new_strides, new_offset);
    }

    qtensor_t select(size_type dim, size_type index) const {
        if (dim >= shape_.size()) {
            throw std::out_of_range("select dimension out of bounds");
        }
        if (index >= shape_[dim]) {
            throw std::out_of_range("select index out of bounds");
        }
        std::vector<size_type> new_shape;
        std::vector<size_type> new_strides;
        new_shape.reserve(shape_.size() - 1);
        new_strides.reserve(strides_.size() - 1);
        for (size_type i = 0; i < shape_.size(); ++i) {
            if (i == dim) {
                continue;
            }
            new_shape.push_back(shape_[i]);
            new_strides.push_back(strides_[i]);
        }
        size_type new_offset = offset_ + index * strides_[dim];
        return qtensor_t(data_, plane_, new_shape, new_strides, new_offset);
    }

    qtensor_t view(
        const std::vector<size_type>& new_shape,
        const std::vector<size_type>& new_strides,
        size_type offset
    ) const {
        if (new_shape.size() != new_strides.size()) {
            throw std::invalid_argument("shape and stride dimensionality mismatch");
        }
        qtensor_t view_tensor(data_, plane_, new_shape, new_strides, offset_ + offset);
        const size_type total = view_tensor.size();
        if (total == 0) {
            return view_tensor;
        }
        size_type max_index = view_tensor.offset_;
        for (size_type i = 0; i < new_shape.size(); ++i) {
            if (new_shape[i] == 0) {
                continue;
            }
            max_index += (new_shape[i] - 1) * new_strides[i];
        }
        if (max_index >= plane_) {
            throw std::out_of_range("view exceeds underlying storage");
        }
        return view_tensor;
    }

    // The elements in row-major order (any layout).
    std::vector<qnumber_type> to_vector() const {
        std::vector<qnumber_type> result(size());
        if (result.empty()) {
            return result;
        }
        size_type out_index = 0;
        for_each_index([&](size_type src) {
            qnumber_type& q = result[out_index++];
            for (int c = 0; c < ncomp; ++c) {
                q.comp(c) = (*data_)[c * plane_ + src];
            }
        });
        return result;
    }

    // A contiguous copy (any layout) -- the planes of the result are the
    // elements in row-major order.
    qtensor_t copy() const {
        qtensor_t out(shape_);
        size_type out_index = 0;
        for_each_index([&](size_type src) {
            for (int c = 0; c < ncomp; ++c) {
                (*out.data_)[c * out.plane_ + out_index] = (*data_)[c * plane_ + src];
            }
            ++out_index;
        });
        return out;
    }

    // The same elements in another scalar type / number of tangents:
    // dropped position tangents are lost, added ones are zero.
    template <typename T2, int NT2>
    qtensor_t<T2, NT2> convert() const {
        qtensor_t<T2, NT2> out(shape_);
        const int nc = std::min(ncomp, qtensor_t<T2, NT2>::ncomp);
        size_type out_index = 0;
        for_each_index([&](size_type src) {
            for (int c = 0; c < nc; ++c) {
                out.plane(c)[out_index] = static_cast<T2>((*data_)[c * plane_ + src]);
            }
            ++out_index;
        });
        return out;
    }

private:
    std::shared_ptr<std::vector<T>> data_;
    size_type plane_ = 0;  // elements per component plane in the storage
    std::vector<size_type> shape_;
    std::vector<size_type> strides_;
    size_type offset_ = 0;

    qtensor_t(
        std::shared_ptr<std::vector<T>> data,
        size_type plane,
        std::vector<size_type> shape,
        std::vector<size_type> strides,
        size_type offset
    )
        : data_(std::move(data)),
          plane_(plane),
          shape_(std::move(shape)),
          strides_(std::move(strides)),
          offset_(offset) {
        check_shape_product(shape_);
    }

    size_type linear_index(const std::vector<size_type>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::out_of_range("index dimensionality mismatch");
        }
        size_type index = offset_;
        for (size_type dim = 0; dim < indices.size(); ++dim) {
            if (indices[dim] >= shape_[dim]) {
                throw std::out_of_range("index out of bounds");
            }
            index += indices[dim] * strides_[dim];
        }
        return index;
    }

    // f(linear index within a plane) over the elements in row-major order
    template <typename F>
    void for_each_index(F f) const {
        if (size() == 0) {
            return;
        }
        if (is_contiguous()) {
            const size_type n = size();
            for (size_type i = 0; i < n; ++i) {
                f(offset_ + i);
            }
            return;
        }
        std::vector<size_type> idx(shape_.size(), 0);
        size_type lin = offset_;
        const size_type nd = shape_.size();
        while (true) {
            f(lin);
            std::ptrdiff_t d = static_cast<std::ptrdiff_t>(nd) - 1;
            for (; d >= 0; --d) {
                const size_type ud = static_cast<size_type>(d);
                lin += strides_[ud];
                if (++idx[ud] < shape_[ud]) {
                    break;
                }
                lin -= strides_[ud] * shape_[ud];
                idx[ud] = 0;
            }
            if (d < 0) {
                break;
            }
        }
    }

    static std::vector<size_type>
    compute_contiguous_strides(const std::vector<size_type>& shape) {
        std::vector<size_type> strides(shape.size(), 1);
        for (std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(shape.size()) - 2;
             idx >= 0; --idx) {
            size_type i = static_cast<size_type>(idx);
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    static size_type
    compute_size(const std::vector<size_type>& shape) noexcept {
        if (shape.empty()) {
            return 0;
        }
        return std::accumulate(
            shape.begin(),
            shape.end(),
            static_cast<size_type>(1),
            [](size_type lhs, size_type rhs) { return lhs * rhs; }
        );
    }

    static void ensure_step_valid(size_type step) {
        if (step == 0) {
            throw std::invalid_argument("slice step must be non-zero");
        }
    }

    void check_shape_product(const std::vector<size_type>& shape) const {
        if (!data_) {
            throw std::runtime_error("qtensor has no underlying storage");
        }
        const size_type required = compute_size(shape);
        if (required == 0) {
            return;
        }
        if (offset_ + required > plane_) {
            throw std::out_of_range("qtensor view exceeds storage bounds");
        }
    }
};

// The Task's tensor: double, four tangents.
using qtensor = qtensor_t<double, 4>;

// A contiguous tensor with the elements of x (x itself, sharing its
// storage, when it already is contiguous).
template <typename T, int NT>
inline qtensor_t<T, NT> contiguous(const qtensor_t<T, NT>& x) {
    if (x.is_contiguous()) {
        return x;
    }
    return x.copy();
}

// Element-wise chain rule: f(v, fv, dfv) sets the value fv and the
// derivative dfv at v; the tangent planes are scaled by dfv.
template <typename T, int NT, typename F>
inline qtensor_t<T, NT> elementwise_chain(const qtensor_t<T, NT>& x, F f) {
    const qtensor_t<T, NT> xc = contiguous(x);
    qtensor_t<T, NT> out(xc.shape());
    const std::size_t n = xc.size();
    const T* v = xc.plane(0);
    T* ov = out.plane(0);
    std::vector<T> d(n);
    for (std::size_t i = 0; i < n; ++i) {
        T fv, dfv;
        f(v[i], fv, dfv);
        ov[i] = fv;
        d[i] = dfv;
    }
    for (int c = 1; c < qtensor_t<T, NT>::ncomp; ++c) {
        const T* t = xc.plane(c);
        T* o = out.plane(c);
        for (std::size_t i = 0; i < n; ++i) {
            o[i] = d[i] * t[i];
        }
    }
    return out;
}

// a + s * b for tensors of the same shape (the residual connection).
template <typename T, int NT>
inline qtensor_t<T, NT> add(
    const qtensor_t<T, NT>& a, const qtensor_t<T, NT>& b, double s = 1.0
) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("add: shape mismatch");
    }
    const qtensor_t<T, NT> ac = contiguous(a);
    const qtensor_t<T, NT> bc = contiguous(b);
    qtensor_t<T, NT> out(ac.shape());
    const std::size_t n = ac.size();
    const T sc = static_cast<T>(s);
    for (int c = 0; c < qtensor_t<T, NT>::ncomp; ++c) {
        const T* pa = ac.plane(c);
        const T* pb = bc.plane(c);
        T* po = out.plane(c);
        for (std::size_t i = 0; i < n; ++i) {
            po[i] = pa[i] + sc * pb[i];
        }
    }
    return out;
}

// x * s with a qnumber s (product rule on every element), e.g. the
// division of an image by its measured brightness: mul(x, 1.0 / norm)
template <typename T, int NT>
inline qtensor_t<T, NT> mul(
    const qtensor_t<T, NT>& x, const qnumber_t<T, NT>& s
) {
    const qtensor_t<T, NT> xc = contiguous(x);
    qtensor_t<T, NT> out(xc.shape());
    const std::size_t n = xc.size();
    const T* v = xc.plane(0);
    T* ov = out.plane(0);
    for (std::size_t i = 0; i < n; ++i) {
        ov[i] = v[i] * s.v;
    }
    for (int c = 1; c < qtensor_t<T, NT>::ncomp; ++c) {
        const T* t = xc.plane(c);
        T* o = out.plane(c);
        const T sc = s.comp(c);
        for (std::size_t i = 0; i < n; ++i) {
            o[i] = t[i] * s.v + v[i] * sc;
        }
    }
    return out;
}

// y += a * x on all components (the inner loop of scalar reductions).
template <typename T, int NT>
inline void axpy(qnumber_t<T, NT>& y, double a, const qnumber_t<T, NT>& x) {
    const T s = static_cast<T>(a);
    y.v += s * x.v;
    y.g1 += s * x.g1;
    y.g2 += s * x.g2;
    if constexpr (NT == 4) {
        y.x1 += s * x.x1;
        y.x2 += s * x.x2;
    }
}

} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_TENSOR_H
