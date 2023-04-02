//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_OPTIMIZER_H
#define NEURALIB_OPTIMIZER_H

#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "layers/Layer.h"

using Eigen::Tensor;


template <class T, class Func>
class Optimizer {
protected:
    Func optimization_step;
public:
    explicit Optimizer(Func optimizationStep) : optimization_step(optimizationStep) {}

    virtual TensorHolder<T> apply_optimization(TensorHolder<T>& gradients) = 0;
};

namespace optimizers{
    template <class T>
    TensorHolder<T> sgd_step(TensorHolder<T> &gradients, T learning_rate){
        if (gradients.size() == 2){
            Tensor<T, 2> grads = gradients.template get<2>();
            Tensor<T, 2> grads_multiplied = grads * grads.constant(learning_rate);
            return TensorHolder<T>(grads_multiplied);
        }
        Tensor<T, 3> grads = gradients.template get<3>();
        Tensor<T, 3> grads_multiplied = grads * grads.constant(learning_rate);
        return TensorHolder<T>(grads_multiplied);
    };

    template <class T>
    class SGD: Optimizer<T, TensorHolder<T> (*)(TensorHolder<T>&, T)> {
        T learning_rate;
    public:
        explicit SGD(T learning_date_): Optimizer<T, TensorHolder<T> (*)(TensorHolder<T>&, T)>(sgd_step),
                learning_rate{learning_date_}{};

        TensorHolder<T> apply_optimization(TensorHolder<T>& gradients) override{
            return this->optimization_step(gradients, learning_rate);
        }
    };
}


#endif //NEURALIB_OPTIMIZER_H
