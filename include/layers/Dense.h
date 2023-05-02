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
#include <map>
#include "utils/cuda/CudaMatrix.cuh"
#include "utils/cuda/matrix_operations.cuh"

using Eigen::Tensor;

template<class T, size_t Dim=2>
class DenseLayer : public Layer<T, Dim> {
    size_t n_in;
    size_t n_hidden;
    Tensor<T, Dim> weights;
    Tensor<T, Dim> biases;
//    Tensor<T, Dim+1> X;
//    tbb::concurrent_queue<Tensor<T, Dim+1>> input_queue;
    std::map<int, Tensor<T, Dim+1>> inputQueue;
//    tbb::concurrent_hash_map<int, Tensor<T, Dim+1>> inputQueue;

public:
    DenseLayer(size_t n_in_, size_t n_hidden_, const std::string& name, Initializer<T>& initializer, bool trainable = true) :
            Layer<T, Dim>(name, trainable),
            n_in(n_in_),
            n_hidden(n_hidden_),
            weights{initializer.get_weights_2d(n_in_, n_hidden_)},
            biases{initializer.get_weights_2d(1, n_hidden_)} {
        biases.setConstant(0);
    };

    Tensor<T, Dim+1> forward(const Tensor<T, Dim+1> & inputs, int minibatchInd = 1, bool train = true) override {
//        std::cout << "[" << n_in << ", " << n_hidden << std::endl;
//        X = inputs;
//        if (train){
//            input_queue.push(inputs);
//        }
        if (train){
            if (inputQueue.find(minibatchInd) != inputQueue.end()){
                inputQueue[minibatchInd] = inputs;
            }
            else{
                inputQueue.template emplace(minibatchInd, inputs);
            }
        }

//        Tensor<T, Dim> output = inputs.reshape(Eigen::array<size_t , 2>{size_t(inputs.dimension(0)), n_in}).contract(
//                weights.shuffle(Eigen::array<int, 2>{1, 0}),
//                Eigen::array<Eigen::IndexPair<int>, 1> {Eigen::IndexPair<int>(1, 0)}
//                ) + biases.broadcast(Eigen::array<size_t, 2>{1, size_t(inputs.dimension(0))}).shuffle(Eigen::array<int, 2>{1, 0});
//
//        return output.reshape(Eigen::array<size_t , 3>{size_t(inputs.dimension(0)), n_hidden, 1});
        return weights
            .contract(inputs, Eigen::array<Eigen::IndexPair<int>, 1> {Eigen::IndexPair<int>(1, 1)})
            .shuffle(Eigen::array<int, 3>{1, 0, 2})
            +
            biases
            .shuffle(Eigen::array<int, 2>{1, 0})
            .broadcast(Eigen::array<size_t, 3>{size_t(inputs.dimension(0)), 1, 1});
    };

    Tensor<T, Dim+1> backward(const Tensor<T, Dim+1> & out_gradient, Optimizer<T> & optimizer, int minibatchInd = 1) override {
//        std::cout << "[" << n_in << ", " << n_hidden << "]: " << out_gradient.mean() << std::endl;
//        if (n_hidden == 10)
//            std::cout << out_gradient << std::endl << std::endl;
//        std::cout << "start of backward" << " " << out_gradient.dimension(0) << " " << out_gradient.dimension(1) << " " << out_gradient.dimension(2) << std::endl;
    Tensor<T, Dim+1> backward(const Tensor<T, Dim+1> & out_gradient, Optimizer<T> & optimizer) override {
        Eigen::array<size_t, 3> reshape_size{size_t(out_gradient.dimension(0)),
                                             size_t(out_gradient.dimension(1)),
                                             size_t(out_gradient.dimension(1))};

        Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = { Eigen::IndexPair<int>(1, 0) };
//        while(!input_queue.try_pop(X));

        for (Eigen::Index i = 0; i < X.dimension(0); ++i) {
            Tensor<T, Dim> weights_gradient = out_gradient.chip(i, 0);
            Tensor<T, Dim> X_chipped = X.chip(i, 0);

            #ifdef CUDA_ENABLE
            CudaMatrix m1 (out_gradient.dimension(1), out_gradient.dimension(2), out_gradient.data());
            CudaMatrix m2 (X.dimension(1), X.dimension(2), X_chipped.data());

            T *res;
            res = (T*) malloc(out_gradient.dimension(1) * X.dimension(2) * sizeof(T));

            matrix_operations::cuda_dense_backward(m1, m2, res, X.dimension(0));

            Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajorBit>> result (res, out_gradient.dimension(1), X.dimension(2));

            #else
            Tensor<T, 2> result =
                    weights_gradient
                            .contract(X_chipped.shuffle(Eigen::array<int, Dim>{1, 0}), contract_dims)
                    /
                    double(X.dimension(0));
            #endif

            weights -= optimizer.apply_optimization(result);
            biases -= optimizer.apply_optimization(Tensor<T, Dim>{out_gradient.chip(i, 0) / double(X.dimension(0))});
        }

//        Tensor<T, Dim> weights_gradient = (out_gradient.broadcast(
//                Eigen::array<size_t , 3>{1, 1, size_t(out_gradient.dimension(1))}
//                ).reshape(reshape_size) * X.template broadcast(Eigen::array<size_t, 3>{1, 1, size_t(X.dimension(1))})
//                        .reshape(reshape_size))
//                        .mean(Eigen::array<int, 1>{0});
//        weights -= optimizer.apply_optimization(weights_gradient);
//        biases -= optimizer.apply_optimization(Tensor<T, Dim>{out_gradient.mean(Eigen::array<int, 1>{0})});
//        std::cout << "before\n";
        Tensor<T, Dim+1> out = out_gradient.template contract(
                weights.shuffle(Eigen::array<int, 2>{1, 0}),
                Eigen::array<Eigen::IndexPair<int>, 1> {Eigen::IndexPair<int>(1, 1)}
                ).reshape(Eigen::array<size_t, 3>({size_t(out_gradient.dimension(0)), size_t(weights.dimension(1)), 1}));
//        std::cout <<"after\n";

        return out;
    };

    void set_weights(const Tensor<T, Dim> & weights_) override {
        weights = std::move(weights_);
    };

    const Tensor<T, Dim> &get_weights() override { return weights; };

    const Tensor<T, Dim> &get_biases() { return biases; };

};

#endif //NEURALIB_DENSE_H
