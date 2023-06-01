//
// Created by Naz on 3/31/Dim0Dim3.
//

//#define CUDA_ENABLE

#ifndef NEURALIB_DENSE_H
#define NEURALIB_DENSE_H

#ifdef CUDA_ENABLE
#include "utils/cuda/matrix_operations.cuh"
#endif
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
    Tensor<T, Dim, Eigen::RowMajor> weights;
    Tensor<T, Dim, Eigen::RowMajor> biases;
    std::mutex mutex;
    tbb::concurrent_unordered_map<int, Tensor<T, Dim+1, Eigen::RowMajor>> input_hash_map;

public:
    DenseLayer(size_t n_in_, size_t n_hidden_, const std::string &name, Initializer<T> &initializer,
               bool trainable = true) : Layer<T, Dim>(name, trainable), n_in(n_in_), n_hidden(n_hidden_),
                                        weights{initializer.get_weights_2d(n_in_, n_hidden_)},
                                        biases{initializer.get_weights_2d(1, n_hidden_)} {
        biases.setConstant(0);
    };

    Tensor<T, Dim+1, Eigen::RowMajor> forward(const Tensor<T, Dim+1, Eigen::RowMajor> &inputs, int minibatchInd = 1, bool train = true) override {
        if (train) {
            auto it = input_hash_map.find(minibatchInd);
            if (it != input_hash_map.end()) {
                input_hash_map[minibatchInd] = inputs;
            }
            else {
                input_hash_map.emplace(minibatchInd, inputs);
            }
        }

        Tensor<T, Dim, Eigen::RowMajor> flattened_input = inputs.reshape(Eigen::array<size_t, 2>{size_t(inputs.dimension(0)), n_in});

#ifdef CUDA_ENABLE
        int rowsA = flattened_input.dimension(0);
        int colsA = flattened_input.dimension(1);
        int colsB = weights.dimension(0);

        Tensor<T, Dim, Eigen::RowMajor> output(rowsA, colsB);

        matrix_operations::cuda_dense_forward(flattened_input.data(), weights.data(), output.data(), rowsA, colsA, colsB, biases.data());
#else
        Tensor<T, Dim, Eigen::RowMajor> output = flattened_input.contract(
                weights.shuffle(Eigen::array<int, 2>{1, 0}),
                Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)})
                        +
                        biases.broadcast(Eigen::array<size_t, 2>{1, size_t(inputs.dimension(0))})
                        .shuffle(Eigen::array<int, 2>{1, 0});
#endif

        return output.reshape(Eigen::array<size_t, 3>{size_t(inputs.dimension(0)), n_hidden, 1});
    };

    Tensor<T, Dim+1, Eigen::RowMajor> backward(const Tensor<T, Dim+1, Eigen::RowMajor> &out_gradient, Optimizer<T> &optimizer, int minibatchInd = 1) override {
        Eigen::array<size_t, 3> reshape_size{size_t(out_gradient.dimension(0)), size_t(out_gradient.dimension(1)),
                                             size_t(out_gradient.dimension(1))};

#ifdef CUDA_ENABLE
        int rowsA = out_gradient.dimension(1);
        int colsA = out_gradient.dimension(2);
        int colsB = input_hash_map[minibatchInd].dimension(1);

        Tensor<T, Dim, Eigen::RowMajor> weights_gradient(rowsA, colsB);
#endif
        for (Eigen::Index i = 0; i < input_hash_map[minibatchInd].dimension(0); ++i) {
            Tensor<T, Dim, Eigen::RowMajor> A = out_gradient.chip(i, 0);
            Tensor<T, Dim, Eigen::RowMajor> B = input_hash_map[minibatchInd].chip(i, 0);
#ifdef CUDA_ENABLE
            matrix_operations::cuda_dense_backward(A.data(), B.data(), weights_gradient.data(), rowsA, colsA, colsB, out_gradient.dimension(0));
#else
            Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};

            Tensor<T, Dim, Eigen::RowMajor> weights_gradient =
                    A.contract(B.shuffle(Eigen::array<int, Dim>{1, 0}), contract_dims) / double(input_hash_map[minibatchInd].dimension(0));
#endif
            const Tensor<T, Dim, Eigen::RowMajor>& weights_substr = optimizer.apply_optimization2d(weights_gradient);
            const Tensor<T, Dim, Eigen::RowMajor>& bias_subsrt = optimizer.apply_optimization2d(
                    Tensor<T, Dim, Eigen::RowMajor>{out_gradient.chip(i, 0) / double(input_hash_map[minibatchInd].dimension(0))});

            mutex.lock();
            weights -= weights_substr;
            biases -= bias_subsrt;
            mutex.unlock();
        }

        Tensor<T, Dim+1, Eigen::RowMajor> out = out_gradient.template contract(weights.shuffle(Eigen::array<int, 2>{1, 0}),
                                                                Eigen::array<Eigen::IndexPair<int>, 1>{
                                                                        Eigen::IndexPair<int>(1, 1)}).reshape(
                Eigen::array<size_t, 3>({size_t(out_gradient.dimension(0)), size_t(weights.dimension(1)), 1}));

        return out;
    };

    void set_weights(const Tensor<T, Dim, Eigen::RowMajor> &weights_) {
        weights = std::move(weights_);
    };

    const Tensor<T, Dim, Eigen::RowMajor> &get_weights() { return weights; };

    const Tensor<T, Dim, Eigen::RowMajor> &get_biases() { return biases; };

    Tensor<T, Dim+1, Eigen::RowMajor> &get_saved_minibatch(int minibatchInd) override {
        auto it = input_hash_map.find(minibatchInd);
        if (it != input_hash_map.end()) {
            return input_hash_map[minibatchInd];
        }
        else {
            throw std::out_of_range ("Minibatch index is out of range!");
        }
    };
};

#endif //NEURALIB_DENSE_H
