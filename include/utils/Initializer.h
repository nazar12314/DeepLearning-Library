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
    TensorHolder<T> get_weights(){return TensorHolder(Tensor<T, 2>());};
};


#endif //NEURALIB_INITIALIZER_H
