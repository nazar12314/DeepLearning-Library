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
class Activation : public Layer<T> {
public:
    Activation(const std::string & name,
               std::function<TensorHolder<T>(const TensorHolder<T> &, std::vector<T> &)> activation_,
               std::function<TensorHolder<T>(const TensorHolder<T> &, std::vector<T> &)> activation_prime_)
               : activation(activation_), activation_prime(activation_prime_),
                 Layer<T>(name, false){};

    void set_weights(const TensorHolder<T> &) override {
        throw std::logic_error("Weights aren't implemented for Activation class");
    }

    const TensorHolder<T> &get_weights() override {
        throw std::logic_error("Weights aren't implemented for Activation class");
    }

    virtual TensorHolder<T> forward(const TensorHolder<T> &inputs) = 0;

    virtual TensorHolder<T> backward(const TensorHolder<T> &out_gradient, Optimizer<T> &optimizer) = 0;

protected:
    std::function<TensorHolder<T>(const TensorHolder<T> &, std::vector<T> &)> activation;
    std::function<TensorHolder<T>(const TensorHolder<T> &, std::vector<T> &)> activation_prime;
};


namespace activations {
    // ReLU
    template<class T>
    TensorHolder<T> relu(const TensorHolder<T> &input, std::vector<T> &) {
        constexpr size_t Dim = 2;
        const Tensor<T, Dim> &input_tensor = input.template get<Dim>();
        Tensor<T, Dim> output_tensor = input_tensor.cwiseMax(Tensor<T, Dim>(input_tensor.dimensions()).setZero());
        return TensorHolder<T>(output_tensor);
    }

    template<class T>
        TensorHolder<T> relu_prime(const TensorHolder<T> &input, std::vector<T> &) {
        constexpr size_t Dim = 2;
        const Tensor<T, Dim> &input_tensor = input.template get<Dim>();
        auto relu_derivative = [](T x) { return x > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0); };
        Tensor<T, Dim> output_tensor = input_tensor.unaryExpr(relu_derivative);
        return TensorHolder<T>(output_tensor);
    }


    template<class T>
    class ReLU : public Activation<T> {
    private:
            std::vector<T> params{};
    public:
        explicit ReLU(const std::string & name) : Activation<T>(name, relu<T>, relu_prime<T>) {}

        TensorHolder<T> forward(const TensorHolder<T> &inputs) override {
            return this->activation(inputs, params);
        }

        TensorHolder<T> backward(const TensorHolder<T> &out_gradient, Optimizer<T> &) override {
            return this->activation_prime(out_gradient, params);
        }

    };

    template<class T>
    TensorHolder<T> sigmoid(const TensorHolder<T> &input, std::vector<T> &) {
        constexpr size_t Dim = 2;
        const Tensor<T, Dim> &input_tensor = input.template get<Dim>();
        const Tensor<T, Dim> &minus_input_tensor = -input_tensor;
        Tensor<T, Dim> output_tensor = input_tensor.constant(1) / (input_tensor.constant(1) + minus_input_tensor.exp());
        return TensorHolder<T>(output_tensor);
    }

    template<class T>
    TensorHolder<T> sigmoid_prime(const TensorHolder<T> &input, std::vector<T> &) {
        constexpr size_t Dim = 2;

        const Tensor<T, Dim> &input_tensor = input.template get<Dim>();
        const Tensor<T, Dim> &minus_input_tensor = -input_tensor;
        Tensor<T, Dim> output_tensor = input_tensor.constant(1) / (input_tensor.constant(1) + minus_input_tensor.exp());

        output_tensor = output_tensor * (output_tensor.constant(1) - output_tensor);
        return TensorHolder<T>(output_tensor);
    }

    template<class T>
    class Sigmoid : public Activation<T> {
    private:
        std::vector<T> params{};
    public:
        explicit Sigmoid(const std::string & name) : Activation<T>(name, sigmoid<T>, sigmoid_prime<T>) {}

        TensorHolder<T> forward(const TensorHolder<T> &inputs) override {
            return this->activation(inputs, params);
        }

        TensorHolder<T> backward(const TensorHolder<T> &out_gradient, Optimizer<T> & optimizer) override {
            return this->activation_prime(out_gradient, params);
        }

    };
}


#endif //NEURALIB_ACTIVATION_H
