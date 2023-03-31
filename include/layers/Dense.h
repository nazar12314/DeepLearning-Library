//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_DENSE_H
#define NEURALIB_DENSE_H

#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "../utils/Initializer.h"
#include "Layer.h"

using Eigen::Tensor;

template <class T, size_t Dim = 2>
class Dense : public Layer<T, Dim> {
    Tensor<T, Dim> weights;
    Tensor<T, Dim> biases;
    Initializer initializer;

public:
    Dense(const std::string &name, bool trainable, const Initializer &initializer);
};

#endif //NEURALIB_DENSE_H
