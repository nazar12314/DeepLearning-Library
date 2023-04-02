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
    RandomNormal(size_t inputs_, size_t outputs_): Initializer<T>(inputs_, outputs_) {};

    TensorHolder<T> get_weights() override {
        Tensor<T, 2> weights(this->inputs, this->outputs);
        weights.template setRandom<Eigen::internal::NormalRandomGenerator<double>>();

        return TensorHolder<T>(weights);
    };

    TensorHolder<T> get_biases() override {
        Tensor<T, 2> biases(this->inputs, 1);
        biases.template setRandom<Eigen::internal::NormalRandomGenerator<double>>();

        return TensorHolder<T>(biases);
    };
};

#endif //NEURALIB_RANDOMNORMAL_H
