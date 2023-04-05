//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_LOSS_H
#define NEURALIB_LOSS_H

#include <iostream>
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "TensorHolder.h"

using Eigen::Tensor;

template<class T>
class Loss{
public:
    Loss(std::function<Tensor<T, 0>(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output)> error_func_,
         std::function<TensorHolder<T>(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output)> error_prime_func_)
            : error_func(error_func_), error_prime_func(error_prime_func_) {};

    explicit Loss(const Loss<T>* loss): error_func{loss->error_func}, error_prime_func{loss->error_prime_func}{}

    Tensor<T, 0> calculate_loss(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output) {
        return this->error_func(pred_output, true_output);
    };

    TensorHolder<T> calculate_grads(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output) {
        return this->error_prime_func(pred_output, true_output);
    };

protected:
    std::function<Tensor<T, 0>(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output)> error_func;
    std::function<TensorHolder<T>(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output)> error_prime_func;
};

namespace loss_functions {
    template<class T>
    Tensor<T, 0> mse_func(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output) {
//        std::cout << "HERE" << std::endl;
        constexpr size_t Dim = 2;
//        std::cout << pred_output.size() << std::endl;
//        std::cout << true_output.size() << std::endl;
        const Tensor<T, Dim> &pred_tensor = pred_output.template get<Dim>();
        const Tensor<T, Dim> &true_tensor = true_output.template get<Dim>();
//        std::cout << pred_tensor.dimension(0) << " " << pred_tensor.dimension(1) << std::endl;
//        std::cout << true_tensor.dimension(0) << " " << true_tensor.dimension(1) << std::endl;
//        std::cout << "HERE2" << std::endl;
        const Tensor<T, 0> error = (pred_tensor - true_tensor).pow(2).mean();
//        std::cout << "HERE3" << std::endl;
        return error;
    }

    template<class T>
    TensorHolder<T> mse_prime_func(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output) {
        constexpr size_t Dim = 2;
        const Tensor<T, Dim> &pred_tensor = pred_output.template get<Dim>();
        const Tensor<T, Dim> &true_tensor = true_output.template get<Dim>();
        Tensor<T, Dim> differ = (pred_tensor-true_tensor);
        Tensor<T, Dim> error = differ*differ.constant(2.0f/differ.dimension(0));
        return TensorHolder<T>(error);
    }

    template<class T>
    class MSE : public Loss<T> {
    public:
        MSE() : Loss<T>(mse_func<T>, mse_prime_func<T>) {}
    };
}

#endif //NEURALIB_LOSS_H
