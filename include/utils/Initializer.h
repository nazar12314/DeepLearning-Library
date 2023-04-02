//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_INITIALIZER_H
#define NEURALIB_INITIALIZER_H

#include "utils/TensorHolder.h"
#include "eigen3/unsupported/Eigen/CXX11/Tensor"

using Eigen::Tensor;

template<class T>
class Initializer {
protected:
    size_t outputs;
    size_t inputs;

public:
    Initializer(size_t inputs_, size_t outputs_): inputs(inputs_), outputs(outputs_) {};

    virtual TensorHolder<T> get_weights() = 0;
    virtual TensorHolder<T> get_biases() = 0;

    virtual ~Initializer() = default;
};


#endif //NEURALIB_INITIALIZER_H
