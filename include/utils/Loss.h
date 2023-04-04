//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_LOSS_H
#define NEURALIB_LOSS_H


#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "TensorHolder.h"

using Eigen::Tensor;

template<class T>
class Loss{
public:
    Loss(std::function<TensorHolder<T>(const TensorHolder<T> &, std::vector<T> &)> error_func_,
         std::function<TensorHolder<T>(const TensorHolder<T> &, std::vector<T> &)> error_prime_func_)
            : error_func(error_func_), error_prime_func(error_prime_func_) {};

    TensorHolder<T> get_error(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output) {
        return this->error_func(pred_output, true_output);
    };

    TensorHolder<T> get_error_der(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output) {
        return this->error_prime_func(pred_output, true_output);
    };

protected:
    std::function<TensorHolder<T>(const TensorHolder<T> &, std::vector<T> &)> error_func;
    std::function<TensorHolder<T>(const TensorHolder<T> &, std::vector<T> &)> error_prime_func;
};

namespace loss_functions {
    template<class T>
    TensorHolder<T> mse_func(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output) {
        constexpr size_t Dim = 2;
        const Tensor<T, Dim> &pred_tensor = pred_output.template get<Dim>();
        const Tensor<T, Dim> &true_tensor = true_output.template get<Dim>();
        Tensor<T, Dim> error = (pred_tensor-true_tensor).pow(2).mean();
        return TensorHolder<T>(error);
    }

    template<class T>
    TensorHolder<T> mse_prime_func(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output) {
        constexpr size_t Dim = 2;
        const Tensor<T, Dim> &pred_tensor = pred_output.template get<Dim>();
        const Tensor<T, Dim> &true_tensor = true_output.template get<Dim>();
        Tensor<T, Dim> differ = (pred_tensor-true_tensor);
        Tensor<T, Dim> error = differ*differ.constant(2/differ.dimension(0));
        return TensorHolder<T>(error);
    }

    template<class T>
    class MSE : public Loss<T> {
    public:
        MSE() : Loss<T>(mse_func<T>, mse_prime_func<T>) {}
    };
}

#endif //NEURALIB_LOSS_H
