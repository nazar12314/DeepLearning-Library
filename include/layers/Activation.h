//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_ACTIVATION_H
#define NEURALIB_ACTIVATION_H

#include <exception>
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "Layer.h"

using Eigen::Tensor;

template<class T, size_t Dim>
class Activation : public Layer<T, Dim> {
public:
    Activation() : Layer<T>("", false) {}

    void set_weights(const Tensor<T, 2> &) override {
        throw std::logic_error("Weights aren't implemented for Activation class");
    }

    void set_weights(const Tensor<T, 3> &) override {
        throw std::logic_error("Weights aren't implemented for Activation class");
    }

    const Tensor<T, 3> &get_weights() override {
        throw std::logic_error("Weights aren't implemented for Activation class");
    }
};


namespace activations {

    template<class T, size_t Dim>
    class ReLU : public Activation<T, Dim> {
    public:
        ReLU() : Activation<T, Dim>(){}

        Tensor<T, Dim> forward(const Tensor<T, Dim> &inputs) override {
            return inputs.cwiseMax(0.0);
        }

        Tensor<T, 3> backward(const Tensor<T, 3> &out_gradient, Optimizer<T> &optimizer) override {
            auto relu_derivative = [](T x) { return x > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0); };
            return out_gradient.unaryExpr(relu_derivative);
        }
    };

    template<class T, size_t Dim>
    class Softmax : public Activation<T, Dim> {
    public:
        Softmax() : Activation<T, Dim>() {}

        Tensor<T, Dim> forward(const Tensor<T, Dim> &inputs) override {
            Tensor<T, Dim> exp_inputs = inputs.exp();
            const Tensor<T, 0> & input_tensor_sum = exp_inputs.sum();
            return exp_inputs / inputs.constant(input_tensor_sum(0));
        }

        Tensor<T, Dim> backward(const Tensor<T, Dim> &out_gradient, Optimizer<T> & optimizer) override {
            return out_gradient;
        }

    };

}


#endif //NEURALIB_ACTIVATION_H