//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_LAYER_H
#define NEURALIB_LAYER_H

#include <iostream>
#include <string>
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils/Optimizer.h"

using Eigen::Tensor;

template<class T, size_t Dim>
class Layer {
    std::string name;
    bool trainable;
public:
    Layer(const std::string & name, bool trainable): name(name), trainable(trainable) {};

    virtual Tensor<T, Dim> forward(const Tensor<T, Dim> & inputs) = 0;

    virtual Tensor<T, Dim> backward(const Tensor<T, Dim> & out_gradient, Optimizer<T>& optimizer) = 0;

    virtual void set_weights(const Tensor<T, Dim> & weights_) = 0;

    virtual const Tensor<T, Dim> &get_weights() = 0;

    virtual ~Layer() = default;
};


#endif //NEURALIB_LAYER_H
