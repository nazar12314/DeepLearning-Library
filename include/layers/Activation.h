//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_ACTIVATION_H
#define NEURALIB_ACTIVATION_H


#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "Layer.h"

using Eigen::Tensor;

template<class T, class Func>
class Activation : Layer<T> {
    Func activation;
    Func activation_prime;
public:
    Activation(const std::string &name, bool trainable, Func activation, Func activationPrime) :
            Layer<T>(name, trainable),
            activation(activation),
            activation_prime(activationPrime) {}

    void set_weights(const TensorHolder<T> &) override = delete;

    const TensorHolder<T> &get_weights() override = delete;

    void adjust_weights(const TensorHolder<T> &) override = delete;
};


#endif //NEURALIB_ACTIVATION_H
