//
// Created by Naz on 3/31/Dim0Dim3.
//

#define CUDA_ENABLE

#ifndef NEURALIB_DENSE_H
#define NEURALIB_DENSE_H

#include "utils/cuda/matrix_operations.cuh"
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

        Tensor<T, Dim, Eigen::RowMajor> output = inputs.reshape(Eigen::array<size_t, 2>{size_t(inputs.dimension(0)), n_in}).contract(
                weights.shuffle(Eigen::array<int, 2>{1, 0}),
                Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)}) +
                                biases.broadcast(Eigen::array<size_t, 2>{1, size_t(inputs.dimension(0))}).shuffle(
                                        Eigen::array<int, 2>{1, 0});

        return output.reshape(Eigen::array<size_t, 3>{size_t(inputs.dimension(0)), n_hidden, 1});
    };

    Tensor<T, Dim+1, Eigen::RowMajor> backward(const Tensor<T, Dim+1, Eigen::RowMajor> &out_gradient, Optimizer<T> &optimizer, int minibatchInd = 1) override {
        Eigen::array<size_t, 3> reshape_size{size_t(out_gradient.dimension(0)), size_t(out_gradient.dimension(1)),
                                             size_t(out_gradient.dimension(1))};

#ifdef CUDA_ENABLE
        int rowsA = out_gradient.dimension(1);
        int colsA = out_gradient.dimension(2);
        int colsB = input_hash_map[minibatchInd].dimension(1);

        T* result = new T[rowsA * colsB];
#endif
        for (Eigen::Index i = 0; i < input_hash_map[minibatchInd].dimension(0); ++i) {
            Tensor<T, Dim, Eigen::RowMajor> A = out_gradient.chip(i, 0);
            Tensor<T, Dim, Eigen::RowMajor> B = input_hash_map[minibatchInd].chip(i, 0).shuffle(Eigen::array<int, Dim>{1, 0});
#ifdef CUDA_ENABLE
            matrix_operations::cuda_dense_backward(A.data(), B.data(), result, rowsA, colsA, colsB, out_gradient.dimension(0));
            Tensor<T, Dim, Eigen::RowMajor> weights_gradient = Eigen::TensorMap<Eigen::Tensor<T, Dim, Eigen::RowMajor>> (result, rowsA, colsB);
#else
            Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};

            Tensor<T, Dim, Eigen::RowMajor> weights_gradient =
                    A.contract(B, contract_dims) / double(input_hash_map[minibatchInd].dimension(0));
#endif

            const Tensor<T, Dim, Eigen::RowMajor>& weights_substr = optimizer.apply_optimization(weights_gradient);
            const Tensor<T, Dim, Eigen::RowMajor>& bias_subsrt = optimizer.apply_optimization(
                    Tensor<T, Dim, Eigen::RowMajor>{out_gradient.chip(i, 0) / double(input_hash_map[minibatchInd].dimension(0))});

            mutex.lock();
            weights -= weights_substr;
            biases -= bias_subsrt;
            mutex.unlock();
        }
#ifdef CUDA_ENABLE
        delete[] result;
#endif
        Tensor<T, Dim+1, Eigen::RowMajor> out = out_gradient.template contract(weights.shuffle(Eigen::array<int, 2>{1, 0}),
                                                                Eigen::array<Eigen::IndexPair<int>, 1>{
                                                                        Eigen::IndexPair<int>(1, 1)}).reshape(
                Eigen::array<size_t, 3>({size_t(out_gradient.dimension(0)), size_t(weights.dimension(1)), 1}));
        return out;
    };

    void set_weights(const Tensor<T, Dim, Eigen::RowMajor> &weights_) override {
        weights = std::move(weights_);
    };

    const Tensor<T, Dim, Eigen::RowMajor> &get_weights() override { return weights; };

    const Tensor<T, Dim, Eigen::RowMajor> &get_biases() { return biases; };

};

#endif //NEURALIB_DENSE_H
