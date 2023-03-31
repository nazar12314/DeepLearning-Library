//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_OPTIMIZER_H
#define NEURALIB_OPTIMIZER_H

#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "../layers/Layer.h"

using Eigen::Tensor;


template <class Func>
class Optimizer {
    Func optimization_step;

public:
    template<class T, size_t Dim = 2>
    const Tensor<T, Dim>& apply_gradient(const Tensor<T, Dim>& gradients, Layer<T, Dim> Layer);
};


#endif //NEURALIB_OPTIMIZER_H
