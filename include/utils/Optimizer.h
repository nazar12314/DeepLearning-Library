//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_OPTIMIZER_H
#define NEURALIB_OPTIMIZER_H

#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils/TensorHolder.h"
#include <iostream>

using Eigen::Tensor;


template<class T>
class Optimizer {
protected:
//    Func optimization_step;
    std::function<TensorHolder<T>(const TensorHolder<T> &, std::vector<T> &)> optimization_step;
public:
    explicit Optimizer(std::function<TensorHolder<T>(const TensorHolder<T> &, std::vector<T> &)> optimizationStep)
            : optimization_step(optimizationStep) {}

    virtual TensorHolder<T> apply_optimization(const TensorHolder<T> &gradients) = 0;
};

namespace optimizers {
    template<class T>
    TensorHolder<T> sgd_step(const TensorHolder<T> &gradients, std::vector<T> &params) {
        // params: learning rate
        T learning_rate = params[0];
        if (gradients.size() == 2) {
            const Tensor<T, 2> &grads = gradients.template get<2>();
            Tensor<T, 2> grads_multiplied = grads * grads.constant(learning_rate);
            return TensorHolder<T>(grads_multiplied);
        }
        const Tensor<T, 3> &grads = gradients.template get<3>();
        Tensor<T, 3> grads_multiplied = grads * grads.constant(learning_rate);
        return TensorHolder<T>(grads_multiplied);
    };

    template<class T>
    class SGD : public Optimizer<T> {
        std::vector<T> params;
    public:
        explicit SGD(T learning_date_) : Optimizer<T>(sgd_step<T>){
            params.template emplace_back(learning_date_);
        }

        TensorHolder<T> apply_optimization(const TensorHolder<T> &gradients) override {
            return this->optimization_step(gradients, params);
        }
    };
}


#endif //NEURALIB_OPTIMIZER_H
