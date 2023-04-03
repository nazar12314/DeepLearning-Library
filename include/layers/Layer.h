//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_LAYER_H
#define NEURALIB_LAYER_H

#include <string>
#include "utils/TensorHolder.h"
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
//#include "utils/Optimizer.h"

using Eigen::Tensor;

template<class T>
class Layer {
    std::string name;
    bool trainable;
public:
    Layer(const std::string & name, bool trainable): name(name), trainable(trainable) {};

    virtual TensorHolder<T> forward(const TensorHolder<T> & inputs) = 0;

    virtual TensorHolder<T> backward(TensorHolder<T> & out_gradient) = 0;

    virtual void set_weights(const TensorHolder<T> & weights_) = 0;

    virtual const TensorHolder<T> &get_weights() = 0;

    virtual void adjust_weights(TensorHolder<T> weights_) = 0;

    virtual void adjust_biases(TensorHolder<T> & biases_) = 0;

    virtual ~Layer() = default;
};


#endif //NEURALIB_LAYER_H
