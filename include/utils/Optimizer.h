//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_OPTIMIZER_H
#define NEURALIB_OPTIMIZER_H

#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "layers/Layer.h"

using Eigen::Tensor;


template <class T>
class Optimizer {
protected:
//    Func optimization_step;
    std::function<TensorHolder<T>(TensorHolder<T>&, std::vector<T>)> optimization_step;
public:
    explicit Optimizer(std::function<TensorHolder<T>(TensorHolder<T>&, std::vector<T>)> optimizationStep) : optimization_step(optimizationStep) {}

    virtual void apply_optimization(TensorHolder<T>& gradients, Layer<T>& layer) = 0;
};

namespace optimizers{
    template <class T>
    TensorHolder<T> sgd_step(TensorHolder<T> &gradients, std::vector<T> params){
        // params: learning rate
        T learning_rate = params[0];
        if (gradients.size() == 2){
            Tensor<T, 2>& grads = gradients.template get<2>();
            Tensor<T, 2> grads_multiplied = grads * grads.constant(learning_rate);
            return TensorHolder<T>(grads_multiplied);
        }
        Tensor<T, 3>& grads = gradients.template get<3>();
        Tensor<T, 3> grads_multiplied = grads * grads.constant(learning_rate);
        return TensorHolder<T>(grads_multiplied);
    };

    template <class T>
    class SGD: Optimizer<T> {
        T learning_rate;
    public:
        explicit SGD(T learning_date_): Optimizer<T>(sgd_step<T>),
                learning_rate{learning_date_}{};

        void apply_optimization(TensorHolder<T>& gradients, Layer<T>& layer) override{
            layer.adjust_weights(this->optimization_step(gradients, std::vector{learning_rate}));
//            return this->optimization_step(gradients, std::vector{learning_rate});
        }
    };
}


#endif //NEURALIB_OPTIMIZER_H
