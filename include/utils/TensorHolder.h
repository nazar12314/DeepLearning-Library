//
// Created by Naz on 4/2/2023.
//

#ifndef NEURALIB_TENSORHOLDER_H
#define NEURALIB_TENSORHOLDER_H


#include <any>
#include <unsupported/Eigen/CXX11/Tensor>

template<typename T>
class TensorHolder {
private:
    std::any _held;
    size_t _size;
public:
    template<int N>
    constexpr explicit TensorHolder(Eigen::Tensor<T, N> tensor) :
            _held{std::move(tensor)},
            _size{N} {}

//    template<int N>
//    constexpr explicit TensorHolder(const Eigen::Tensor<T, N>& tensor) :
//            _held{tensor},
//            _size{N} {}

    constexpr TensorHolder(const TensorHolder &) = default;

    constexpr TensorHolder(TensorHolder &&) noexcept = default;

    template<size_t N>
    Eigen::Tensor<T, N> &get() {
        return std::any_cast<Eigen::Tensor<T, N> &>(_held);
    }

    template<size_t N>
    const Eigen::Tensor<T, N> &get() const {
        return std::any_cast<const Eigen::Tensor<T, N> &>(_held);
    }

    [[nodiscard]] constexpr int size() const noexcept {
        return _size;
    }

    TensorHolder& operator=(const TensorHolder& other) {
        _held = other._held;
        _size = other._size;
        return *this;
    }

    TensorHolder& operator=(TensorHolder&& other)  noexcept {
        _held = other._held;
        _size = other._size;
        return *this;
    }
};

#endif //NEURALIB_TENSORHOLDER_H
