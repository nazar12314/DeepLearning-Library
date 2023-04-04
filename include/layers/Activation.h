//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_ACTIVATION_H
#define NEURALIB_ACTIVATION_H

#include <exception>
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "Layer.h"
#include "utils/TensorHolder.h"

using Eigen::Tensor;

template<class T>
class Activation : Layer<T> {
public:
    Activation() : Layer<T>("", false) {}

    virtual void set_weights(const TensorHolder<T> &) = 0;

    virtual const TensorHolder<T> &get_weights() = 0;

    virtual TensorHolder<T> forward(const TensorHolder<T> &inputs) = 0;

    virtual TensorHolder<T> backward(const TensorHolder<T> &out_gradient, Optimizer<T> &optimizer) = 0;

protected:
    virtual TensorHolder<T> activation(const TensorHolder<T> &input, const std::vector<T> &) = 0;

    virtual TensorHolder<T> activation_prime(const TensorHolder<T> &input, const std::vector<T> &) = 0;
};


namespace activations {
    // ReLU
    template<class T>
    TensorHolder<T> relu_function(const TensorHolder<T> &input, const std::vector<T> &) {
        constexpr auto Dim = input.size();
        const Tensor<T, Dim> &input_tensor = input.template get<Dim>();
        Tensor<T, Dim> output_tensor = input_tensor.cwiseMax(Tensor<T, Dim>::Zero());
        return TensorHolder<T>(output_tensor);
    }

    template<class T>
    TensorHolder<T> relu_function_prime(const TensorHolder<T> &input, const std::vector<T> &) {
        constexpr auto Dim = input.size();
        const Tensor<T, Dim> &input_tensor = input.template get<Dim>();
        Tensor<T, Dim> output_tensor = input_tensor.cwiseSign().cwiseMax(Tensor<T, Dim>::Zero());
        return TensorHolder<T>(output_tensor);
    }


    template<class T>
    class ReLU : public Activation<T> {
    public:
        ReLU() : Activation<T>() {}

        void set_weights(const TensorHolder<T> &) override {
            throw std::logic_error("Weights aren't implemented for Activation class");
        }

        const TensorHolder<T> &get_weights() override {
            throw std::logic_error("Weights aren't implemented for Activation class");
        }

        TensorHolder<T> forward(const TensorHolder<T> &inputs) override {
            return this->activation(inputs, std::vector<T>());
        }

        TensorHolder<T> backward(const TensorHolder<T> &out_gradient, Optimizer<T> &optimizer) override {
            return this->activation_prime(out_gradient, std::vector<T>());
        }

        TensorHolder<T> activation(const TensorHolder<T> &input, const std::vector<T> &) override {
            constexpr size_t Dim = 2;
            const Tensor<T, Dim> &input_tensor = input.template get<Dim>();
            Tensor<T, Dim> output_tensor = input_tensor.cwiseMax(Tensor<T, Dim>(input_tensor.dimensions()).setZero());
            return TensorHolder<T>(output_tensor);
        }

        TensorHolder<T> activation_prime(const TensorHolder<T> &input, const std::vector<T> &) override {
            constexpr size_t Dim = 2;
            const Tensor<T, Dim> &input_tensor = input.template get<Dim>();
            auto relu_derivative = [](T x) { return x > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0); };
            Tensor<T, Dim> output_tensor = input_tensor.unaryExpr(relu_derivative);
            return TensorHolder<T>(output_tensor);
        }

    };

}


#endif //NEURALIB_ACTIVATION_H
