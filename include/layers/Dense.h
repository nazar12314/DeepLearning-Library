//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_DENSE_H
#define NEURALIB_DENSE_H

#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils/Initializer.h"
#include "Layer.h"

using Eigen::Tensor;

template <class T, size_t Dim = 2>
class DenseLayer : public Layer<T, Dim> {
    Tensor<T, Dim> weights;
    Tensor<T, Dim> biases;
    Initializer initializer;

public:
    DenseLayer(const std::string &name, bool trainable, const Initializer &initializer):
        Layer<T, Dim>(name, trainable), initializer(initializer) {};
    void forward(const Tensor<T, Dim> &) override {};
    Tensor<T, Dim> backward(const Tensor<T, Dim> &) override {return Tensor<T, Dim>();};
    void set_weights(const Tensor<T, Dim> &) override {};
    const Tensor<T, Dim> &get_weights() override {return Tensor<T, Dim>();};
    void adjust_weights(const Tensor<T, Dim> &) override {};
};

#endif //NEURALIB_DENSE_H
