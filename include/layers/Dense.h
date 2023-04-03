//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_DENSE_H
#define NEURALIB_DENSE_H

#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils/Initializer.h"
#include "Layer.h"
#include "utils/Optimizer.h"

using Eigen::Tensor;

template<class T>
class DenseLayer : public Layer<T> {
    size_t n_in;
    size_t n_hidden;
    TensorHolder<T> weights;
    TensorHolder<T> biases;
    Tensor<double, 2> X;;

public:
    DenseLayer(size_t n_in_, size_t n_hidden_, const std::string& name, Initializer<T>& initializer, bool trainable = true) :
            Layer<T>(name, trainable),
            n_in(n_in_),
            n_hidden(n_hidden_),
            weights{initializer.get_weights(n_in_, n_hidden_)},
            biases{initializer.get_weights(1, n_hidden_)}{};

    TensorHolder<T> forward(const TensorHolder<T> & inputs) override {
        X = inputs.template get<2>();
        Tensor<T, 2>& weights_tensor = weights.template get<2>();

        if (weights_tensor.dimension(1) != X.dimension(0)) {
            throw std::invalid_argument("Incompatible tensor shapes");
        }

        Tensor<T, 2> output_tensor = weights_tensor.contract(
                X,
                Eigen::array<Eigen::IndexPair<int>, 1> {Eigen::IndexPair<int>(1, 0)}
                ) + biases.template get<2>();

        return TensorHolder(output_tensor);
    };

    TensorHolder<T> backward(const TensorHolder<T> & out_gradient, Optimizer<T>& optimizer) override {
        Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = { Eigen::IndexPair<int>(1, 0) };
        Tensor<T, 2>& weights_tensor = weights.template get<2>();
        Tensor<T, 2>& biases_tensor = biases.template get<2>();
        const Tensor<T, 2>& out_gradient_tensor = out_gradient.template get<2>();

        Tensor<T, 2> weights_gradient = out_gradient_tensor.contract(X.shuffle(Eigen::array<int, 2>{1, 0}), contract_dims);
        weights_tensor -= optimizer.apply_optimization(TensorHolder<T>(weights_gradient)).template get<2>();
        biases_tensor -= optimizer.apply_optimization(out_gradient).template get<2>();
        Tensor<T, 2> output_tensor = weights_tensor.shuffle(Eigen::array<int, 2>{1, 0}).contract(
                out_gradient_tensor,
                contract_dims);

        return TensorHolder(output_tensor);
    };

    void set_weights(const TensorHolder<T> & weights_) override {
        weights = std::move(weights_);
    };

    const TensorHolder<T> &get_weights() override { return weights; };

};

#endif //NEURALIB_DENSE_H
