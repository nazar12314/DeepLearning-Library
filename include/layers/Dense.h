//
// Created by Naz on 3/31/Dim0Dim3.
//

#ifndef NEURALIB_DENSE_H
#define NEURALIB_DENSE_H

#include "unsupported/Eigen/CXX11/Tensor"
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
    Tensor<T, Dim+1> X;

public:
    DenseLayer(size_t n_in_, size_t n_hidden_, const std::string& name, Initializer<T>& initializer, bool trainable = true) :
            Layer<T, Dim>(name, trainable),
            n_in(n_in_),
            n_hidden(n_hidden_),
            weights{initializer.get_weights_2d(n_in_, n_hidden_)},
            biases{initializer.get_weights_2d(1, n_hidden_)} {};

    Tensor<T, Dim+1> forward(const Tensor<T, Dim+1> & inputs) override {
        X = inputs;

//        if (weights.dimension(1) != X.dimension(0)) {
//            throw std::invalid_argument("Incompatible tensor shapes");
//        }

        Tensor<T, Dim> output = inputs.reshape(Eigen::array<size_t , 2>{size_t(inputs.dimension(0)), n_in}).contract(
                weights.shuffle(Eigen::array<int, 2>{1, 0}),
                Eigen::array<Eigen::IndexPair<int>, 1> {Eigen::IndexPair<int>(1, 0)}
                ) + biases.broadcast(Eigen::array<size_t, 2>{1, size_t(inputs.dimension(0))}).shuffle(Eigen::array<int, 2>{1, 0});

        return output.reshape(Eigen::array<size_t , 3>{size_t(inputs.dimension(0)), n_hidden, 1});
    };

    Tensor<T, Dim+1> backward(const Tensor<T, Dim+1> & out_gradient, Optimizer<T> & optimizer) override {
//        std::cout << "start of backward" << " " << out_gradient.dimension(0) << " " << out_gradient.dimension(1) << " " << out_gradient.dimension(2) << std::endl;
        Eigen::array<size_t, 3> reshape_size{size_t(out_gradient.dimension(0)),
                                             size_t(out_gradient.dimension(1)),
                                             size_t(out_gradient.dimension(1))};

        Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = { Eigen::IndexPair<int>(1, 0) };
        for (Eigen::Index i = 0; i < X.dimension(0); ++i) {
            Tensor<T, Dim> weights_gradient = out_gradient.chip(i, 0).contract(
                    X.chip(i, 0).shuffle(Eigen::array<int, Dim>{1, 0}),
                    contract_dims) / double(X.dimension(0));
            weights -= optimizer.apply_optimization(weights_gradient);
            biases -= optimizer.apply_optimization(Tensor<T, Dim>{out_gradient.chip(i, 0) / double(X.dimension(0))});
        }


//        Tensor<T, Dim> weights_gradient = (out_gradient.broadcast(
//                Eigen::array<size_t , 3>{1, 1, size_t(out_gradient.dimension(1))}
//                ).reshape(reshape_size) * X.template broadcast(Eigen::array<size_t, 3>{1, 1, size_t(X.dimension(1))})
//                        .reshape(reshape_size))
//                        .mean(Eigen::array<int, 1>{0});
//        weights -= optimizer.apply_optimization(weights_gradient);
//        biases -= optimizer.apply_optimization(Tensor<T, Dim>{out_gradient.mean(Eigen::array<int, 1>{0})});

        Tensor<T, Dim+1> out = out_gradient.template contract(
                weights.shuffle(Eigen::array<int, 2>{1, 0}),
                Eigen::array<Eigen::IndexPair<int>, 1> {Eigen::IndexPair<int>(1, 1)}
                ).reshape(Eigen::array<size_t, 3>({size_t(out_gradient.dimension(0)), size_t(weights.dimension(1)), 1}));


//        Tensor<T, Dim+1> out(X.dimension(0), X.dimension(1), X.dimension(2));
//        for (Eigen::Index i = 0; i < X.dimension(0); ++i) {
////            std::cout << i << std::endl;
//            out.chip(i, 0) = weights.shuffle(Eigen::array<int, 2>{1, 0}).contract(out_gradient.chip(i, 0), contract_dims);
//        }
        return out;
    };

    void set_weights(const Tensor<T, Dim> & weights_) override {
        weights = std::move(weights_);
    };

    const Tensor<T, Dim> &get_weights() override { return weights; };

};

#endif //NEURALIB_DENSE_H
