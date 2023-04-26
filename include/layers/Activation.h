//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_ACTIVATION_H
#define NEURALIB_ACTIVATION_H

#include <exception>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Layer.h"

using Eigen::Tensor;

template<class T, size_t Dim>
class Activation : public Layer<T, Dim> {
public:
    Activation() : Layer<T, Dim>("", false) {}

    void set_weights(const Tensor<T, Dim> & weights_) override {
        throw std::logic_error("Weights aren't implemented for Activation class");
    }

    const Tensor<T, Dim> &get_weights() override {
        throw std::logic_error("Weights aren't implemented for Activation class");
    }
};


namespace activations {

    template<class T, size_t Dim>
    class ReLU : public Activation<T, Dim> {
    public:
        ReLU() : Activation<T, Dim>(){}

        Tensor<T, Dim+1> forward(const Tensor<T, Dim+1> &inputs) override {
            return inputs.cwiseMax(0.0);
        }

        Tensor<T, Dim+1> backward(const Tensor<T, Dim+1> &out_gradient, Optimizer<T> &optimizer) override {
            auto relu_derivative = [](T x) { return x > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0); };
            return out_gradient.unaryExpr(relu_derivative);
        }
    };

    template<class T, size_t Dim>
    class Softmax : public Activation<T, Dim> {
    public:
        Tensor<T, Dim> output;
        Softmax() : Activation<T, Dim>() {}

        Tensor<T, Dim> forward(const Tensor<T, Dim> &inputs) override {
            Tensor<T, Dim> exp_inputs = inputs.exp();
            Tensor<T, 0> input_tensor_sum = exp_inputs.sum();

            Tensor<T, Dim> tmp_output = exp_inputs / inputs.constant(input_tensor_sum(0));

            size_t output_size = tmp_output.size();
            Eigen::DSizes<long, Dim> output_dims = tmp_output.dimensions();

            Eigen::array<long, Dim> new_shape {output_dims[0], output_dims[1], output_dims[2]};
            Eigen::TensorMap<Eigen::Tensor<long, Dim>> reshaped_output(tmp_output.data(), new_shape);

            Eigen::array<long, Dim> broadcast_shape { 1, 1, output_size };

            output = reshaped_output
                    .broadcast(broadcast_shape);

            return tmp_output;
        }

        Tensor<T, Dim> backward(const Tensor<T, Dim> &out_gradient, Optimizer<T> & optimizer) override {
            Eigen::Tensor<long, Dim> transposed = output.shuffle(Eigen::array<int, 3>{0, 2, 1});

            Eigen::MatrixXd m = Eigen::MatrixXd::Identity(output.dimension(1), output.dimension(1));

            Eigen::TensorMap<const Tensor<double, Dim>> t(m.data(), 1, output.dimension(1), output.dimension(1));

            Tensor<double, Dim> identity = t.broadcast(Eigen::array<int, 3>({output.dimension(1), 1, 1}));

            Eigen::Tensor<long, Dim> result = output * (identity - transposed);

            Eigen::Tensor<long, Dim> broad_grads = out_gradient.broadcast(Eigen::array<int, 3>(1, 1, result.dimension(2)));

            result *= broad_grads.shuffle(Eigen::array<int, 3>{0, 2, 1});

            return result.sum(Eigen::array<long, 1>{2}).reshape(Eigen::array<long, 3>{output.dimension(0), output.dimension(2), 1});
        }
    };
}


#endif //NEURALIB_ACTIVATION_H