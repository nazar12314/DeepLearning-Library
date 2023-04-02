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
    Activation(Func activation, Func activationPrime) :
            Layer<T>("", false),
            activation(activation),
            activation_prime(activationPrime) {}

    void set_weights(const TensorHolder<T> &) override{};

    const TensorHolder<T> &get_weights() override{ return TensorHolder<T>(Tensor<T, 0>());};

    void adjust_weights(const TensorHolder<T> &) override{};

    TensorHolder<T> forward(const TensorHolder<T> & inputs) {

    };

    TensorHolder<T> backward(const TensorHolder<T> & inputs) {

    };

};


namespace activations {
    template<class T>
    T relu_func(T x) {
        return std::max(0, x);
    }

    template<class T>
    T relu_func_prime(T x) {
        return (x > 0) ? 1 : 0;
    }

    template<class T>
    class ReLU : Activation<T, T(T)> {
    public:
        ReLU() : Activation<T, T(T)>(relu_func, relu_func_prime) {}
    };
}


#endif //NEURALIB_ACTIVATION_H
