//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_ACTIVATION_H
#define NEURALIB_ACTIVATION_H


#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "Layer.h"

using Eigen::Tensor;

template<class T>
class Activation : Layer<T> {
protected:
    std::function<TensorHolder<T>(TensorHolder<T> &, std::vector<T>)> activation;
    std::function<TensorHolder<T>(TensorHolder<T> &, std::vector<T>)> activation_prime;
public:
    Activation(std::function<TensorHolder<T>(TensorHolder<T> &, std::vector<T>)> activation,
               std::function<TensorHolder<T>(TensorHolder<T> &, std::vector<T>)> activationPrime) :
            Layer<T>("", false),
            activation(activation),
            activation_prime(activationPrime) {}

    void set_weights(const TensorHolder<T> &) override {};

    const TensorHolder<T> &get_weights() override { return TensorHolder<T>(Tensor<T, 0>()); };

    void adjust_weights(const TensorHolder<T> &) override {};

    TensorHolder<T> forward(const TensorHolder<T> &inputs) override {
        return activation(inputs);
    };

    TensorHolder<T> backward(const TensorHolder<T> &inputs) override {
        return activation_prime(inputs);
    };

};


namespace activations {

    template<typename T = double, size_t Dim = 2>
    TensorHolder<T> relu_function(TensorHolder<T> &input) {
        TensorHolder<T> output = input;
        Tensor<T, Dim> &input_tensor = input.template get<Dim>();
        Tensor<T, Dim> &output_tensor = output.template get<Dim>();
        output_tensor = input_tensor.unaryExpr([](T x) { return std::max(x, static_cast<T>(0)); });
        return output;
    }

    template<class T = double, size_t Dim = 2>
    TensorHolder<T> relu_function_prime(TensorHolder<T> &input) {
        TensorHolder<T> output = input;
        Tensor<T, Dim> &input_tensor = input.template get<Dim>();
        Tensor<T, Dim> &output_tensor = output.template get<Dim>();
        output_tensor = input_tensor.unaryExpr(
                [](T x) { return (x > static_cast<T>(0)) ? static_cast<T>(1) : static_cast<T>(0); });
        return output;
    }

    template<class T>
    class ReLU : public Activation<T> {
    public:
        ReLU() : Activation<T>(relu_function<T>, relu_function_prime<T>) {}
    };
}


#endif //NEURALIB_ACTIVATION_H
