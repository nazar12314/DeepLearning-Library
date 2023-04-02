//
// Created by Nazar Kononenko on 02.04.2023.
//

#ifndef NEURALIB_RANDOMNORMAL_H
#define NEURALIB_RANDOMNORMAL_H

#include "utils/TensorHolder.h"
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils/Initializer.h"

template<class T>
class RandomNormal: public Initializer<T> {
public:
    TensorHolder<T> get_weights(size_t n_in, size_t n_hidden) override {
        Tensor<T, 2> weights(n_hidden, n_in);
        weights.template setRandom<Eigen::internal::NormalRandomGenerator<double>>();

        return TensorHolder<T>(weights);
    };

    TensorHolder<T> get_biases(size_t n_hidden) override {
        Tensor<T, 2> biases(n_hidden, 1);
        biases.template setRandom<Eigen::internal::NormalRandomGenerator<double>>();

        return TensorHolder<T>(biases);
    };
};

#endif //NEURALIB_RANDOMNORMAL_H
