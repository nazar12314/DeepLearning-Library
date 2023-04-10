//
// Created by Naz on 3/31/Dim0Dim3.
//

#ifndef NEURALIB_DENSE_H
#define NEURALIB_DENSE_H

#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils/Initializer.h"
#include "Layer.h"
#include "utils/Optimizer.h"

using Eigen::Tensor;

template<class T, size_t Dim=2>
class DenseLayer : public Layer<T, Dim> {
    size_t n_in;
    size_t n_hidden;
    Tensor<T, Dim> weights;
    Tensor<T, Dim> biases;
    Tensor<T, Dim> X;

public:
    DenseLayer(size_t n_in_, size_t n_hidden_, const std::string& name, Initializer<T>& initializer, bool trainable = true) :
            Layer<T, Dim>(name, trainable),
            n_in(n_in_),
            n_hidden(n_hidden_),
            weights{initializer.get_weights_2d(n_in_, n_hidden_)},
            biases{initializer.get_weights_2d(1, n_hidden_)} {};

    Tensor<T, Dim> forward(const Tensor<T, Dim> & inputs) override {
        X = inputs;

        if (weights.dimension(1) != X.dimension(0)) {
            throw std::invalid_argument("Incompatible tensor shapes");
        }

        Tensor<T, Dim> output_tensor = weights.contract(
                X,
                Eigen::array<Eigen::IndexPair<int>, 1> {Eigen::IndexPair<int>(1, 0)}
                ) + biases;

        return output_tensor;
    };

    Tensor<T, Dim> backward(const Tensor<T, Dim> & out_gradient, Optimizer<T> & optimizer) override {
        Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = { Eigen::IndexPair<int>(1, 0) };

        Tensor<T, Dim> weights_gradient = out_gradient.contract(X.shuffle(Eigen::array<int, Dim>{1, 0}), contract_dims);
        weights -= optimizer.apply_optimization(weights_gradient);
        biases -= optimizer.apply_optimization(out_gradient);

        Tensor<T, Dim> output_tensor = weights.shuffle(Eigen::array<int, Dim>{1, 0}).contract(
                out_gradient,
                contract_dims);

        return output_tensor;
    };

    void set_weights(const Tensor<T, Dim> & weights_) override {
        weights = std::move(weights_);
    };

    const Tensor<T, Dim> &get_weights() override { return weights; };

};

#endif //NEURALIB_DENSE_H
