//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_OPTIMIZER_H
#define NEURALIB_OPTIMIZER_H

#include "unsupported/Eigen/CXX11/Tensor"
#include <iostream>

using Eigen::Tensor;


template<class T>
class Optimizer {
protected:
public:
    explicit Optimizer() = default;

    virtual Tensor<T, 2, Eigen::RowMajor> apply_optimization(const Tensor<T, 2, Eigen::RowMajor> &gradients) = 0;

    virtual Tensor<T, 3, Eigen::RowMajor> apply_optimization(const Tensor<T, 3, Eigen::RowMajor> &gradients) = 0;


};

namespace optimizers {
    template<class T>
    class SGD : public Optimizer<T> {
        T learning_rate;
    public:
        explicit SGD(T learning_rate_) : Optimizer<T>(), learning_rate(learning_rate_){}

        Tensor<T, 2, Eigen::RowMajor> apply_optimization(const Tensor<T, 2, Eigen::RowMajor> &gradients) override {
            Tensor<T, 2, Eigen::RowMajor> grads_multiplied = gradients * gradients.constant(learning_rate);
            return grads_multiplied;
        }

        Tensor<T, 3, Eigen::RowMajor> apply_optimization(const Tensor<T, 3, Eigen::RowMajor> &gradients) override {
            Tensor<T, 3, Eigen::RowMajor> grads_multiplied = gradients * gradients.constant(learning_rate);
            return gradients * gradients.constant(learning_rate);
        }
    };
}


#endif //NEURALIB_OPTIMIZER_H