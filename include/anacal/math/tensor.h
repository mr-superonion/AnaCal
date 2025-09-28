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

namespace anacal {
namespace math {

class qtensor {
public:
    using size_type = std::size_t;

    qtensor()
        : data_(std::make_shared<std::vector<math::qnumber>>()) {}

    explicit qtensor(const std::vector<size_type>& shape)
        : data_(std::make_shared<std::vector<math::qnumber>>(compute_size(shape))),
          shape_(shape),
          strides_(compute_contiguous_strides(shape)),
          offset_(0) {}

    qtensor(
        const std::vector<size_type>& shape,
        const math::qnumber& fill_value
    )
        : data_(std::make_shared<std::vector<math::qnumber>>(
              compute_size(shape), fill_value
          )),
          shape_(shape),
          strides_(compute_contiguous_strides(shape)),
          offset_(0) {}

    qtensor(const qtensor&) = default;
    qtensor(qtensor&&) noexcept = default;
    qtensor& operator=(const qtensor&) = default;
    qtensor& operator=(qtensor&&) noexcept = default;
    ~qtensor() = default;

    static qtensor from_shape(const std::vector<size_type>& shape) {
        return qtensor(shape);
    }

    static qtensor from_shape(
        const std::vector<size_type>& shape,
        const math::qnumber& fill_value
    ) {
        return qtensor(shape, fill_value);
    }

    static qtensor from_vector(
        std::vector<math::qnumber>&& data,
        const std::vector<size_type>& shape
    ) {
        const size_type expected = compute_size(shape);
        if (data.size() != expected) {
            std::ostringstream ss;
            ss << "qtensor::from_vector expects " << expected
               << " elements but received " << data.size();
            throw std::invalid_argument(ss.str());
        }
        auto storage = std::make_shared<std::vector<math::qnumber>>(
            std::move(data)
        );
        return qtensor(
            std::move(storage),
            shape,
            compute_contiguous_strides(shape),
            0
        );
    }

    static qtensor from_vector(
        const std::vector<math::qnumber>& data,
        const std::vector<size_type>& shape
    ) {
        const size_type expected = compute_size(shape);
        if (data.size() != expected) {
            std::ostringstream ss;
            ss << "qtensor::from_vector expects " << expected
               << " elements but received " << data.size();
            throw std::invalid_argument(ss.str());
        }
        auto storage = std::make_shared<std::vector<math::qnumber>>(data);
        return qtensor(
            std::move(storage),
            shape,
            compute_contiguous_strides(shape),
            0
        );
    }

    static qtensor from_image(
        std::vector<math::qnumber>&& data,
        size_type height,
        size_type width
    ) {
        return from_vector(std::move(data), {height, width});
    }

    size_type ndim() const noexcept { return shape_.size(); }

    const std::vector<size_type>& shape() const noexcept { return shape_; }

    const std::vector<size_type>& strides() const noexcept { return strides_; }

    size_type size() const noexcept { return compute_size(shape_); }

    bool empty() const noexcept { return size() == 0; }

    math::qnumber& at(const std::vector<size_type>& indices) {
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
        return (*data_)[index];
    }

    const math::qnumber& at(const std::vector<size_type>& indices) const {
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
        return (*data_)[index];
    }

    math::qnumber& operator[](size_type index) {
        if (!is_contiguous()) {
            throw std::runtime_error("operator[] requires contiguous storage");
        }
        if (index >= size()) {
            throw std::out_of_range("index out of bounds");
        }
        return (*data_)[offset_ + index];
    }

    const math::qnumber& operator[](size_type index) const {
        if (!is_contiguous()) {
            throw std::runtime_error("operator[] requires contiguous storage");
        }
        if (index >= size()) {
            throw std::out_of_range("index out of bounds");
        }
        return (*data_)[offset_ + index];
    }

    math::qnumber* data() {
        if (!is_contiguous()) {
            throw std::runtime_error("data() requires contiguous storage");
        }
        return data_->data() + offset_;
    }

    const math::qnumber* data() const {
        if (!is_contiguous()) {
            throw std::runtime_error("data() requires contiguous storage");
        }
        return data_->data() + offset_;
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

    qtensor reshape(const std::vector<size_type>& new_shape) const {
        if (!is_contiguous()) {
            throw std::runtime_error("reshape requires contiguous storage");
        }
        if (compute_size(new_shape) != size()) {
            throw std::invalid_argument("reshape size mismatch");
        }
        return qtensor(
            data_,
            new_shape,
            compute_contiguous_strides(new_shape),
            offset_
        );
    }

    qtensor slice(
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
        return qtensor(data_, new_shape, new_strides, new_offset);
    }

    qtensor select(size_type dim, size_type index) const {
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
        return qtensor(data_, new_shape, new_strides, new_offset);
    }

    qtensor view(
        const std::vector<size_type>& new_shape,
        const std::vector<size_type>& new_strides,
        size_type offset
    ) const {
        if (new_shape.size() != new_strides.size()) {
            throw std::invalid_argument("shape and stride dimensionality mismatch");
        }
        qtensor view_tensor(data_, new_shape, new_strides, offset_ + offset);
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
        if (max_index >= view_tensor.data_->size()) {
            throw std::out_of_range("view exceeds underlying storage");
        }
        return view_tensor;
    }

    std::vector<math::qnumber> to_vector() const {
        std::vector<math::qnumber> result(size());
        if (result.empty()) {
            return result;
        }
        size_type out_index = 0;
        copy_recursive(0, offset_, result, out_index);
        return result;
    }

private:
    std::shared_ptr<std::vector<math::qnumber>> data_;
    std::vector<size_type> shape_;
    std::vector<size_type> strides_;
    size_type offset_ = 0;

    qtensor(
        std::shared_ptr<std::vector<math::qnumber>> data,
        std::vector<size_type> shape,
        std::vector<size_type> strides,
        size_type offset
    )
        : data_(std::move(data)),
          shape_(std::move(shape)),
          strides_(std::move(strides)),
          offset_(offset) {
        check_shape_product(shape_);
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
        const size_type max_possible = data_->size();
        if (offset_ + required > max_possible) {
            throw std::out_of_range("qtensor view exceeds storage bounds");
        }
    }

    void copy_recursive(
        size_type dim,
        size_type base_offset,
        std::vector<math::qnumber>& out,
        size_type& out_index
    ) const {
        if (dim == shape_.size()) {
            if (out_index >= out.size()) {
                throw std::out_of_range("copy_recursive overflow");
            }
            out[out_index++] = (*data_)[base_offset];
            return;
        }
        const size_type extent = shape_[dim];
        if (extent == 0) {
            return;
        }
        const size_type stride = strides_[dim];
        for (size_type i = 0; i < extent; ++i) {
            copy_recursive(dim + 1, base_offset + i * stride, out, out_index);
        }
    }
};

} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_TENSOR_H
