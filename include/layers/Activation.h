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
class Activation : Layer<T, Dim> {
public:
    Activation() : Layer<T, Dim>("", false) {}

    void set_weights(const Tensor<T, Dim> &) override {
        throw std::logic_error("Weights aren't implemented for Activation class");
    }

    const Tensor<T, Dim> &get_weights() override {
        throw std::logic_error("Weights aren't implemented for Activation class");
    }

    virtual Tensor<T, Dim> forward(const Tensor<T, Dim> &inputs) = 0;

    virtual Tensor<T, Dim> backward(const Tensor<T, Dim> &out_gradient, Optimizer<T> &optimizer) = 0;

protected:
    virtual Tensor<T, Dim> activation(const Tensor<T, Dim> &input, const std::vector<T> &) = 0;

    virtual Tensor<T, Dim> activation_prime(const Tensor<T, Dim> &input, const std::vector<T> &) = 0;
};


namespace activations {

    template<class T, size_t Dim>
    class ReLU : public Activation<T, Dim> {
    public:
        ReLU() : Activation<T, Dim>() {}

        Tensor<T, Dim> forward(const Tensor<T, Dim> &inputs) override {
            return this->activation(inputs, std::vector<T>());
        }

        Tensor<T, Dim> backward(const Tensor<T, Dim> &out_gradient, Optimizer<T> &optimizer) override {
            return this->activation_prime(out_gradient, std::vector<T>());
        }

        Tensor<T, Dim> activation(const Tensor<T, Dim> &input, const std::vector<T> &) override {
            return input.cwiseMax(0.0);
        }

        Tensor<T, Dim> activation_prime(const Tensor<T, Dim> &input, const std::vector<T> &) override {
            auto relu_derivative = [](T x) { return x > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0); };
            return input.unaryExpr(relu_derivative);
        }

    };

}


#endif //NEURALIB_ACTIVATION_H