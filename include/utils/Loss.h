//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_ACTIVATION_H
#define NEURALIB_ACTIVATION_H


#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "TensorHolder.h"

using Eigen::Tensor;

template<class T>
class Loss{
public:

protected:
    virtual TensorHolder<T> error_func(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output) = 0;

    virtual TensorHolder<T> error_prime_func(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output) = 0;
};


namespace activations {

    template<class T>
    class MSE : public Loss<T> {
    public:
        MSE() : Loss<T>() {}

        TensorHolder<T> error_func(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output) override {
            constexpr auto real_dim = pred_output.size();
            assert(realDim == true_output.size());
            const Tensor<T, real_dim> &pred_tensor = pred_output.template get<real_dim>();
            const Tensor<T, real_dim> &true_tensor = true_output.template get<real_dim>();
            Tensor<T, real_dim> error = (pred_tensor-true_tensor).pow(2).mean();
            return TensorHolder<T>(error);
        }

        TensorHolder<T> error_prime_func(const TensorHolder<T> &pred_output, const TensorHolder<T> &true_output) override {
            constexpr auto real_dim = pred_output.size();
            assert(realDim == true_output.size());
            const Tensor<T, real_dim> &pred_tensor = pred_output.template get<real_dim>();
            const Tensor<T, real_dim> &true_tensor = true_output.template get<real_dim>();
            Tensor<T, real_dim> differ = (pred_tensor-true_tensor);
            Tensor<T, real_dim> error = differ*differ.constant(2/real_dim);
            return TensorHolder<T>(error);
        }
    };
}

#endif //NEURALIB_ACTIVATION_H
