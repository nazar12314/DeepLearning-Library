//
// Created by Naz on 4/2/2023.
//

#ifndef NEURALIB_TENSOR_HOLDER_H
#define NEURALIB_TENSOR_HOLDER_H


#include <any>
#include <unsupported/Eigen/CXX11/Tensor>

template<typename T>
class tensor_holder {
private:
    std::any _held;
    size_t _size;
public:
    template<int N>
    constexpr tensor_holder(Eigen::Tensor<T, N> tensor) :
            _held{std::move(tensor)},
            _size{N} {}

    constexpr tensor_holder(const tensor_holder &) = default;

    constexpr tensor_holder(tensor_holder &&) = default;

    template<size_t N>
    Eigen::Tensor<T, N> &get() {
        return std::any_cast<Eigen::Tensor<T, N> &>(_held);
    }

    template<size_t N>
    const Eigen::Tensor<T, N> &get() const {
        return std::any_cast<Eigen::Tensor<T, N> &>(_held);
    }

    constexpr int size() const noexcept {
        return _size;
    }
};


#endif //NEURALIB_TENSOR_HOLDER_H
