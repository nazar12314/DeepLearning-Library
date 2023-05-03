//
// Created by Naz on 3/31/Dim0Dim3.
//

#ifndef NEURALIB_DENSE_H
#define NEURALIB_DENSE_H

#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/Initializer.h"
#include "Layer.h"
#include "utils/Optimizer.h"
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>
#include <map>
#include <mutex>


using Eigen::Tensor;

template<class T, size_t Dim = 2>
class DenseLayer : public Layer<T, Dim> {
    size_t n_in;
    size_t n_hidden;
    Tensor<T, Dim> weights;
    Tensor<T, Dim> biases;
    tbb::concurrent_unordered_map<int, Tensor<T, Dim + 1>> input_hash_map;
    std::mutex mutex;

public:
    DenseLayer(size_t n_in_, size_t n_hidden_, const std::string &name, Initializer<T> &initializer,
               bool trainable = true) : Layer<T, Dim>(name, trainable), n_in(n_in_), n_hidden(n_hidden_),
                                        weights{initializer.get_weights_2d(n_in_, n_hidden_)},
                                        biases{initializer.get_weights_2d(1, n_hidden_)},
                                        mutex{}{
        biases.setConstant(0);
    };

    Tensor<T, Dim + 1> forward(const Tensor<T, Dim + 1> &inputs, int minibatchInd = 1, bool train = true) override {

        if (train) {
            auto it = input_hash_map.find(minibatchInd);
            if (it != input_hash_map.end()) {
                input_hash_map[minibatchInd] = inputs;
            }
            else {
                input_hash_map.emplace(minibatchInd, inputs);
            }
        }

        Tensor<T, Dim> output = inputs.reshape(Eigen::array<size_t, 2>{size_t(inputs.dimension(0)), n_in}).contract(
                weights.shuffle(Eigen::array<int, 2>{1, 0}),
                Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)}) +
                                biases.broadcast(Eigen::array<size_t, 2>{1, size_t(inputs.dimension(0))}).shuffle(
                                        Eigen::array<int, 2>{1, 0});

        return output.reshape(Eigen::array<size_t, 3>{size_t(inputs.dimension(0)), n_hidden, 1});
    };

    Tensor<T, Dim + 1>
    backward(const Tensor<T, Dim + 1> &out_gradient, Optimizer<T> &optimizer, int minibatchInd = 1) override {
        Eigen::array<size_t, 3> reshape_size{size_t(out_gradient.dimension(0)), size_t(out_gradient.dimension(1)),
                                             size_t(out_gradient.dimension(1))};

        Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};

        for (Eigen::Index i = 0; i < input_hash_map[minibatchInd].dimension(0); ++i) {
            Tensor<T, Dim> weights_gradient =
                    out_gradient.chip(i, 0).contract(input_hash_map[minibatchInd].chip(i, 0).shuffle(Eigen::array<int, Dim>{1, 0}),
                                                     contract_dims) / double(input_hash_map[minibatchInd].dimension(0));
            const Tensor<T, 2>& weights_substr = optimizer.apply_optimization(weights_gradient);
            const Tensor<T, 2>& bias_subsrt = optimizer.apply_optimization(
                    Tensor<T, Dim>{out_gradient.chip(i, 0) / double(input_hash_map[minibatchInd].dimension(0))});
            mutex.lock();
            weights -= weights_substr;
            biases -= bias_subsrt;
            mutex.unlock();
        }
        Tensor<T, Dim + 1> out = out_gradient.template contract(weights.shuffle(Eigen::array<int, 2>{1, 0}),
                                                                Eigen::array<Eigen::IndexPair<int>, 1>{
                                                                        Eigen::IndexPair<int>(1, 1)}).reshape(
                Eigen::array<size_t, 3>({size_t(out_gradient.dimension(0)), size_t(weights.dimension(1)), 1}));
        return out;
    };

    void set_weights(const Tensor<T, Dim> &weights_) override {
        weights = std::move(weights_);
    };

    const Tensor<T, Dim> &get_weights() override { return weights; };

    const Tensor<T, Dim> &get_biases() { return biases; };

};

#endif //NEURALIB_DENSE_H
