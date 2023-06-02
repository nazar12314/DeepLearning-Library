//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_LOSS_H
#define NEURALIB_LOSS_H

#include <iostream>
#include "unsupported/Eigen/CXX11/Tensor"

using Eigen::Tensor;

template<class T>
class Loss {
public:
    explicit Loss() = default;

    /**
     * @brief Calculate the loss value.
     * @param pred_output The predicted output tensor.
     * @param true_output The true output tensor.
     * @return The calculated loss value.
     */
    virtual Tensor<T, 0, Eigen::RowMajor>
    calculate_loss(const Eigen::Tensor<T, 3, Eigen::RowMajor> &pred_output, const Eigen::Tensor<T, 3, Eigen::RowMajor> &true_output) = 0;

    /**
     * @brief Calculate the gradients of the loss.
     * @param pred_output The predicted output tensor.
     * @param true_output The true output tensor.
     * @return The calculated gradients of the loss.
     */
    virtual Tensor<T, 3, Eigen::RowMajor>
    calculate_grads(const Eigen::Tensor<T, 3, Eigen::RowMajor> &pred_output, const Eigen::Tensor<T, 3, Eigen::RowMajor> &true_output) = 0;

};

namespace loss_functions {
    /**
     * @class MSE
     * @brief Mean Squared Error loss function.
     * @tparam T The data type of the loss values.
     */
    template<class T>
    class MSE : public Loss<T> {
    public:
        MSE() : Loss<T>() {}

        Tensor<T, 0, Eigen::RowMajor>
        calculate_loss(const Eigen::Tensor<T, 3, Eigen::RowMajor> &pred_output, const Eigen::Tensor<T, 3, Eigen::RowMajor> &true_output) override {
            const Tensor<T, 0, Eigen::RowMajor> error = (pred_output - true_output).pow(2).mean();
            return error;
        }

        Tensor<T, 3, Eigen::RowMajor>
        calculate_grads(const Eigen::Tensor<T, 3, Eigen::RowMajor> &pred_output, const Eigen::Tensor<T, 3, Eigen::RowMajor> &true_output) override {
            Tensor<T, 3, Eigen::RowMajor> differ = (pred_output - true_output);
            const Tensor<T, 3, Eigen::RowMajor> error = differ * differ.constant(2.0f / differ.dimension(1));
            return error;
        }
    };

    /**
     * @class BinaryCrossEntropy
     * @brief Binary Cross Entropy loss function.
     * @tparam T The data type of the loss values.
     */
    template<class T>
    class BinaryCrossEntropy : public Loss<T> {
    public:
        BinaryCrossEntropy() : Loss<T>() {}

        Tensor<T, 0, Eigen::RowMajor>
        calculate_loss(const Eigen::Tensor<T, 3, Eigen::RowMajor> &pred_output, const Eigen::Tensor<T, 3, Eigen::RowMajor> &true_output) override {
            const T epsilon = 1e-7;
            const Tensor<T, 0, Eigen::RowMajor> error = (true_output * ((pred_output + pred_output.constant(epsilon)).log()) +
                                        ((true_output.constant(1.0) - true_output) *
                                         ((pred_output.constant(1.0) - pred_output +
                                           pred_output.constant(epsilon)).log()))).mean();
            return -error;
        }

        Tensor<T, 3, Eigen::RowMajor>
        calculate_grads(const Eigen::Tensor<T, 3, Eigen::RowMajor> &pred_output, const Eigen::Tensor<T, 3, Eigen::RowMajor> &true_output) override {
            const T epsilon = 1e-7;
            const Tensor<T, 3, Eigen::RowMajor> error = -(true_output / (pred_output + pred_output.constant(epsilon))) +
                                       ((pred_output.constant(1.0) - true_output) /
                                        (pred_output.constant(1.0) - pred_output + pred_output.constant(epsilon)));
            return error;
        }
    };
}

#endif //NEURALIB_LOSS_H
