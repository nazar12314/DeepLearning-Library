//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_LAYER_H
#define NEURALIB_LAYER_H

#include <string>
#include "eigen3/unsupported/Eigen/CXX11/Tensor"

using Eigen::Tensor;

template<class T, size_t Dim>
class Layer {
    std::string name;
    bool trainable;
    Tensor<T, Dim> inputs;
    Tensor<T, Dim> outputs;

public:

    Layer(const std::string &name, bool trainable): name(name), trainable(trainable) {}


    virtual void forward(const Tensor<T, Dim> &) = 0;

    virtual Tensor<T, Dim> backward(const Tensor<T, Dim> &) = 0;

    virtual void set_weights(const Tensor<T, Dim> &) = 0;

    virtual const Tensor<T, Dim> &get_weights() = 0;

    virtual void adjust_weights(const Tensor<T, Dim> &) = 0;

    virtual ~Layer() = default;
};


#endif //NEURALIB_LAYER_H
