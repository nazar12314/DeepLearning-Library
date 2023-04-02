//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_DENSE_H
#define NEURALIB_DENSE_H

#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils/Initializer.h"
#include "Layer.h"

using Eigen::Tensor;

template<class T>
class DenseLayer : public Layer<T> {
    TensorHolder<T> weights;
    TensorHolder<T> biases;
    TensorHolder<T> inputs;
    Initializer<T> initializer;

public:
    DenseLayer(const std::string &name, bool trainable, const Initializer<T> &initializer_) :
            Layer<T>(name, trainable), initializer(initializer_), weights{initializer.get_weights()},
            biases{initializer.get_weights()}, inputs{TensorHolder<T>(Tensor<T, 2>())}{
    };

    void forward(const TensorHolder<T> &) override {};

    TensorHolder<T> backward(const TensorHolder<T> &) override { return TensorHolder(Tensor<double, 2>()); };

    void set_weights(const TensorHolder<T> &) override {};

    const TensorHolder<T> &get_weights() override { return weights; };

    void adjust_weights(const TensorHolder<T> &) override {};
};

#endif //NEURALIB_DENSE_H
