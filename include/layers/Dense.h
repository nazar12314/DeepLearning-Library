//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_DENSE_H
#define NEURALIB_DENSE_H

#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils/Initializer.h"
#include "Layer.h"
#include "utils/RandomNormal.h"

using Eigen::Tensor;

template<class T>
class DenseLayer : public Layer<T> {
    size_t n_in;
    size_t n_hidden;
    TensorHolder<T> weights;
    TensorHolder<T> biases;
    TensorHolder<T> X;

public:
    DenseLayer(size_t n_in_, size_t n_hidden_, const std::string& name, Initializer<T>& initializer, bool trainable = true) :
            Layer<T>(name, trainable),
            n_in(n_in_),
            n_hidden(n_hidden_),
            weights{initializer.get_weights(n_in_, n_hidden_)},
            biases{initializer.get_weights(1, n_hidden_)},
            X{TensorHolder<T>(Tensor<T, 2>())} {
    };

    TensorHolder<T> forward(const TensorHolder<T> & inputs) override {
        X = std::move(inputs);
        Tensor<T, 2> X_tensor = X.template get<2>();
        Tensor<T, 2> weights_tensor = weights.template get<2>();
        Tensor<T, 2> biases_tensor = biases.template get<2>();

        if (weights_tensor.dimension(1) != X_tensor.dimension(0)) {
            throw std::invalid_argument("Incompatible tensor shapes");
        }

        Tensor<T, 2> output_tensor = weights_tensor.contract(
                X_tensor,
                Eigen::array<Eigen::IndexPair<int>, 1> {Eigen::IndexPair<int>(1, 0)}
                ) + biases_tensor;

        return TensorHolder(output_tensor);
    };

    TensorHolder<T> backward(TensorHolder<T> & out_gradient) override {
        Tensor<T, 2> out_gradient_tensor = out_gradient.template get<2>();
        Tensor<T, 2> weights_tensor = weights.template get<2>();
        Tensor<T, 2> weights_tensor_T = weights_tensor.shuffle(Eigen::array<int, 2>{1, 0});

        Tensor<T, 2> output_tensor = weights_tensor_T.contract(
                out_gradient_tensor,
                Eigen::array<Eigen::IndexPair<int>, 1> {Eigen::IndexPair<int>(1, 0)}
                );

        return TensorHolder(output_tensor);
    };

    void set_weights(const TensorHolder<T> & weights_) override {
        weights = std::move(weights_);
    };

    const TensorHolder<T> &get_weights() override { return weights; };

    void adjust_weights(TensorHolder<T> & other_weights) override {
        Tensor<T, 2> other_weights_tensor = other_weights.template get<2>();
        Tensor<T, 2> weights_tensor = weights.template get<2>();

        Tensor<T, 2> result = weights_tensor - other_weights_tensor;

        weights = std::move(TensorHolder<T>(result));
    };

    void adjust_biases(TensorHolder<T> & other_biases) override {
        Tensor<T, 2> other_biases_tensor = other_biases.template get<2>();
        Tensor<T, 2> biases_tensor = biases.template get<2>();

        Tensor<T, 2> result = biases_tensor - other_biases_tensor;

        biases = std::move(TensorHolder<T>(result));
    };
};

#endif //NEURALIB_DENSE_H
