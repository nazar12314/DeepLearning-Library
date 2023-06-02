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
    bool trainable;
public:
    std::string name;

    /**
     * Constructor for the Layer class.
     *
     * @param name The name of the layer.
     * @param trainable Flag indicating whether the layer is trainable.
     */
    Layer(const std::string & name, bool trainable): name(name), trainable(trainable) {};

    /**
     * Perform the forward pass of the layer.
     *
     * @param inputs The input tensor.
     * @param minibatchInd The minibatch index.
     * @param train Flag indicating whether it's training mode.
     * @return The output tensor.
     */
    virtual Tensor<T, Dim+1, Eigen::RowMajor> forward(const Tensor<T, Dim+1, Eigen::RowMajor> & inputs, int minibatchInd = 1, bool train = true) = 0;

    /**
     * Perform the backward pass of the layer.
     *
     * @param out_gradient The output gradient tensor.
     * @param optimizer The optimizer for weight and bias updates.
     * @param minibatchInd The minibatch index.
     * @return The input gradient tensor.
     */
    virtual Tensor<T, Dim+1, Eigen::RowMajor> backward(const Tensor<T, Dim+1, Eigen::RowMajor> & out_gradient, Optimizer<T>& optimizer, int minibatchInd = 1) = 0;

    /**
     * Get the saved input minibatch tensor for a given minibatch index.
     *
     * @param minibatchInd The minibatch index.
     * @return The input minibatch tensor.
     */
    virtual Tensor<T, Dim+1, Eigen::RowMajor> &get_saved_minibatch(int minibatchInd) = 0;

    /**
     * Virtual destructor for the Layer class.
     */
    virtual ~Layer() = default;
};


#endif //NEURALIB_LAYER_H
