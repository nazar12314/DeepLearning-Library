//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_LOSS_H
#define NEURALIB_LOSS_H

#include <iostream>
#include "unsupported/Eigen/CXX11/Tensor"
//#include "TensorHolder.h"

using Eigen::Tensor;

template<class T>
class Loss {
public:
    explicit Loss() = default;

    virtual Tensor<T, 0> calculate_loss(const Eigen::Tensor<T, 2> &pred_output, const Eigen::Tensor<T, 2> &true_output) = 0;

    virtual Tensor<T, 2> calculate_grads(const Eigen::Tensor<T, 2> &pred_output, const Eigen::Tensor<T, 2> &true_output) = 0;

    virtual Tensor<T, 0> calculate_loss(const Eigen::Tensor<T, 3> &pred_output, const Eigen::Tensor<T, 3> &true_output) = 0;

    virtual Tensor<T, 3> calculate_grads(const Eigen::Tensor<T, 3> &pred_output, const Eigen::Tensor<T, 3> &true_output) = 0;

};

namespace loss_functions {
    template<class T>
    class MSE : public Loss<T> {
    public:
        MSE() : Loss<T>(){}

        Tensor<T, 0> calculate_loss(const Eigen::Tensor<T, 2> &pred_output, const Eigen::Tensor<T, 2> &true_output) override{
            const Tensor<T, 0> error = (pred_output - true_output).pow(2).mean();
            return error;
        }

        Tensor<T, 2> calculate_grads(const Eigen::Tensor<T, 2> &pred_output, const Eigen::Tensor<T, 2> &true_output) override{
            Tensor<T, 2> differ = (pred_output-true_output);
            const Tensor<T, 2> error = differ*differ.constant(2.0f/differ.dimension(0));
            return error;
        }

        Tensor<T, 0> calculate_loss(const Eigen::Tensor<T, 3> &pred_output, const Eigen::Tensor<T, 3> &true_output) override{
            const Tensor<T, 0> error = (pred_output - true_output).pow(2).mean();
            return error;
        }

        Tensor<T, 3> calculate_grads(const Eigen::Tensor<T, 3> &pred_output, const Eigen::Tensor<T, 3> &true_output) override{
            Tensor<T, 3> differ = (pred_output-true_output);
            const Tensor<T, 3> error = differ*differ.constant(2.0f/differ.dimension(1));
            return error;
        }
    };

    template<class T>
    class BinaryCrossEntropy : public Loss<T> {
    public:
        BinaryCrossEntropy() : Loss<T>(){}

        Tensor<T, 0> calculate_loss(const Eigen::Tensor<T, 2> &pred_output, const Eigen::Tensor<T, 2> &true_output) override{
            const T epsilon = 1e-7;
            const Tensor<T, 0> error = (true_output*((pred_output+pred_output.constant(epsilon)).log()) +
                                             ((true_output.constant(1.0) - true_output) * ((pred_output.constant(1.0) - pred_output + pred_output.constant(epsilon)).log())))
                                                     .mean();
            return -error;
        }

        Tensor<T, 2> calculate_grads(const Eigen::Tensor<T, 2> &pred_output, const Eigen::Tensor<T, 2> &true_output) override{
            const T epsilon = 1e-7;
            const Tensor<T, 2> error = -(true_output/(pred_output+pred_output.constant(epsilon))) + ((pred_output.constant(1.0)-true_output)/(pred_output.constant(1.0)-pred_output+pred_output.constant(epsilon)));
            return error;
        }

        Tensor<T, 0> calculate_loss(const Eigen::Tensor<T, 3> &pred_output, const Eigen::Tensor<T, 3> &true_output) override{
            const T epsilon = 1e-7;
            const Tensor<T, 0> error = (true_output*((pred_output+pred_output.constant(epsilon)).log()) +
                                             ((true_output.constant(1.0) - true_output) * ((pred_output.constant(1.0) - pred_output + pred_output.constant(epsilon)).log())))
                    .mean();
            return -error;
        }

        Tensor<T, 3> calculate_grads(const Eigen::Tensor<T, 3> &pred_output, const Eigen::Tensor<T, 3> &true_output) override{
            const T epsilon = 1e-7;
            const Tensor<T, 3> error = -(true_output/(pred_output+pred_output.constant(epsilon))) + ((pred_output.constant(1.0)-true_output)/(pred_output.constant(1.0)-pred_output+pred_output.constant(epsilon)));
            return error;
        }
    };
}

#endif //NEURALIB_LOSS_H
