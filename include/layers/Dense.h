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
    Initializer<T> initializer;
    TensorHolder<T> weights;
    TensorHolder<T> biases;
    TensorHolder<T> X;

public:
    DenseLayer(const std::string &name, bool trainable, const Initializer<T> &initializer_) :
            Layer<T>(name, trainable), initializer(initializer_), weights{initializer.get_weights()},
            biases{initializer.get_weights()}, X{TensorHolder<T>(Tensor<T, 2>())} {
    };

    TensorHolder<T> forward(const TensorHolder<T> & inputs) override {
        X = std::move(inputs);
        Tensor<T, 2> X_tensor = X.template get<2>();
        Tensor<T, 2> weights_tensor = weights.template get<2>();
        Tensor<T, 2> output_tensor = weights_tensor.contract(X_tensor, { {1, 0} });

        return TensorHolder<T> (output_tensor + biases);
    };

    TensorHolder<T> backward(const TensorHolder<T> & out_gradient) override {
        Tensor<T, 2> out_gradient_tensor = out_gradient.template get<2>();
        Tensor<T, 2> weights_tensor = weights.template get<2>();

        return
    };

    void set_weights(const TensorHolder<T> &) override {};

    const TensorHolder<T> &get_weights() override { return weights; };

    void adjust_weights(const TensorHolder<T> &) override {};
};

#endif //NEURALIB_DENSE_H
