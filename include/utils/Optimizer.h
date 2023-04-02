//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_OPTIMIZER_H
#define NEURALIB_OPTIMIZER_H

#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "../layers/Layer.h"

using Eigen::Tensor;


template <class T, class Func>
class Optimizer {
    Func optimization_step;

public:
    explicit Optimizer(Func optimizationStep) : optimization_step(optimizationStep) {}

    virtual const TensorHolder<T>& apply_gradient(const TensorHolder<T>& gradients, Layer<T> Layer) = 0;
};


#endif //NEURALIB_OPTIMIZER_H
