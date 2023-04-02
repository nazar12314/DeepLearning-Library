//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_LAYER_H
#define NEURALIB_LAYER_H

#include <string>
#include "utils/TensorHolder.h"
#include "eigen3/unsupported/Eigen/CXX11/Tensor"

using Eigen::Tensor;

template<class T>
class Layer {
    std::string name;
    bool trainable;
    TensorHolder<T> inputs;

public:

    Layer(const std::string &name, bool trainable): name(name), trainable(trainable) {}

    virtual void forward(const TensorHolder<T> &) = 0;

    virtual TensorHolder<T> backward(const TensorHolder<T> &) = 0;

    virtual void set_weights(const TensorHolder<T> &) = 0;

    virtual const TensorHolder<T> &get_weights() = 0;

    virtual void adjust_weights(const TensorHolder<T> &) = 0;

    virtual ~Layer() = default;
};


#endif //NEURALIB_LAYER_H
