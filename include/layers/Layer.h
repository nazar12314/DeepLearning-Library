//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_LAYER_H
#define NEURALIB_LAYER_H

#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include "utils/Optimizer.h"

using Eigen::Tensor;

template<class T, size_t Dim>
class Layer {
    std::string name;
    bool trainable;
public:
    Layer(const std::string & name, bool trainable): name(name), trainable(trainable) {};

    virtual Tensor<T, Dim+1, Eigen::RowMajor> forward(const Tensor<T, Dim+1, Eigen::RowMajor> & inputs, int minibatchInd = 1, bool train = true) = 0;

    virtual Tensor<T, Dim+1, Eigen::RowMajor> backward(const Tensor<T, Dim+1, Eigen::RowMajor> & out_gradient, Optimizer<T>& optimizer, int minibatchInd = 1) = 0;

//    virtual void set_weights(const Tensor<T, Dim, Eigen::RowMajor> & weights_) = 0;

//    virtual const Tensor<T, Dim, Eigen::RowMajor> &get_weights() = 0;

    virtual Tensor<T, Dim+1, Eigen::RowMajor> &get_saved_minibatch(int minibatchInd) = 0;

    virtual ~Layer() = default;
};


#endif //NEURALIB_LAYER_H
