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

public:

    virtual TensorHolder<T> get_weights(size_t n_in, size_t n_hidden) = 0;
    virtual TensorHolder<T> get_biases(size_t n_hidden) = 0;
//
    virtual ~Initializer() = default;
};


#endif //NEURALIB_INITIALIZER_H
