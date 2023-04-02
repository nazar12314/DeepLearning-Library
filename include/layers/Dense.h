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
    Initializer initializer;

public:
    DenseLayer(const std::string &name, bool trainable, const Initializer &initializer) :
            Layer<T>(name, trainable), initializer(initializer) {};

    void forward(const TensorHolder<T> &) override {};

    TensorHolder<T> backward(const TensorHolder<T> &) override { return TensorHolder<T>(); };

    void set_weights(const TensorHolder<T> &) override {};

    const TensorHolder<T> &get_weights() override { return TensorHolder<T>(); };

    void adjust_weights(const TensorHolder<T> &) override {};
};

#endif //NEURALIB_DENSE_H
