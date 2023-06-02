//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_OPTIMIZER_H
#define NEURALIB_OPTIMIZER_H

#include "unsupported/Eigen/CXX11/Tensor"
#include <iostream>

using Eigen::Tensor;

/**
 * @class Optimizer
 * @brief Base class for optimization algorithms.
 * @tparam T The data type of the optimization parameters.
 */
template<class T>
class Optimizer {
protected:
public:
    explicit Optimizer() = default;

    /**
     * @brief Apply the optimization algorithm to update the 2D parameter tensor.
     * @param gradients The gradients of the parameters.
     * @return The updated parameter tensor.
     */
    virtual Tensor<T, 2, Eigen::RowMajor> apply_optimization2d(const Tensor<T, 2, Eigen::RowMajor> &gradients) = 0;

    /**
     * @brief Apply the optimization algorithm to update the 3D parameter tensor.
     * @param gradients The gradients of the parameters.
     * @return The updated parameter tensor.
     */
    virtual Tensor<T, 3, Eigen::RowMajor> apply_optimization3d(const Tensor<T, 3, Eigen::RowMajor> &gradients) = 0;

    /**
     * @brief Apply the optimization algorithm to update the 4D parameter tensor.
     * @param gradients The gradients of the parameters.
     * @return The updated parameter tensor.
     */
    virtual Tensor<T, 4, Eigen::RowMajor> apply_optimization4d(const Tensor<T, 4, Eigen::RowMajor> &gradients) = 0;
};

namespace optimizers {
    /**
     * @class SGD
     * @brief Stochastic Gradient Descent optimization algorithm.
     * @tparam T The data type of the optimization parameters.
     */
    template<class T>
    class SGD : public Optimizer<T> {
        T learning_rate;
    public:
        /**
         * @brief Construct an SGD optimizer.
         * @param learning_rate_ The learning rate of the optimizer.
         */
        explicit SGD(T learning_rate_) : Optimizer<T>(), learning_rate(learning_rate_){}

        Tensor<T, 2, Eigen::RowMajor> apply_optimization2d(const Tensor<T, 2, Eigen::RowMajor> &gradients) override {
            Tensor<T, 2, Eigen::RowMajor> grads_multiplied = gradients * gradients.constant(learning_rate);
            return grads_multiplied;
        }

        Tensor<T, 3, Eigen::RowMajor> apply_optimization3d(const Tensor<T, 3, Eigen::RowMajor> &gradients) override {
            Tensor<T, 3, Eigen::RowMajor> grads_multiplied = gradients * gradients.constant(learning_rate);
            return gradients * gradients.constant(learning_rate);
        }

        Tensor<T, 4, Eigen::RowMajor> apply_optimization4d(const Tensor<T, 4, Eigen::RowMajor> &gradients) override {
            Tensor<T, 4, Eigen::RowMajor> grads_multiplied = gradients * gradients.constant(learning_rate);
            return gradients * gradients.constant(learning_rate);
        }
    };
}


#endif //NEURALIB_OPTIMIZER_H