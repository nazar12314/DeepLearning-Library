//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_ACTIVATION_H
#define NEURALIB_ACTIVATION_H

#include <exception>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Layer.h"
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>


using Eigen::Tensor;

template<class T, size_t Dim>
class Activation : public Layer<T, Dim> {
public:
    Activation() : Layer<T, Dim>("", false) {}

    void set_weights(const Tensor<T, Dim> &weights_) override {
        throw std::logic_error("Weights aren't implemented for Activation class");
    }

    const Tensor<T, Dim> &get_weights() override {
        throw std::logic_error("Weights aren't implemented for Activation class");
    }
};


namespace activations {

    template<class T, size_t Dim>
    class ReLU : public Activation<T, Dim> {
        tbb::concurrent_unordered_map<int, Tensor<T, Dim + 1>> input_map;

    public:
        ReLU() : Activation<T, Dim>() {}

        Tensor<T, Dim + 1>
        forward(const Tensor<T, Dim + 1> &inputs_, int minibatchInd = 1, bool train = true) override {

            if (train) {
                auto it = input_map.find(minibatchInd);
                if (it != input_map.end()) {
                    input_map[minibatchInd] = inputs_;
                }
                else {
                    input_map.emplace(minibatchInd, inputs_);
                }
            }
            return inputs_.cwiseMax(0.0);
        }

        Tensor<T, Dim + 1>
        backward(const Tensor<T, Dim + 1> &out_gradient, Optimizer<T> &optimizer, int minibatchInd = 1) override {
            auto relu_derivative = [](T x) { return x > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0); };
            return out_gradient * input_map[minibatchInd].unaryExpr(relu_derivative);
        }
    };

    template<class T, size_t Dim>
    class Sigmoid : public Activation<T, Dim> {
        tbb::concurrent_unordered_map<int, Tensor<T, Dim + 1>> input_map;
    public:
        Sigmoid() : Activation<T, Dim>() {}

        Tensor<T, Dim + 1>
        forward(const Tensor<T, Dim + 1> &inputs_, int minibatchInd = 1, bool train = true) override {

            auto res = inputs_.sigmoid();
            if (train) {
                auto it = input_map.find(minibatchInd);
                if (it != input_map.end()) {
                    input_map[minibatchInd] = res;
                }
                else {
                    input_map.emplace(minibatchInd, res);
                }
            }
            return res;
        }

        Tensor<T, Dim + 1>
        backward(const Tensor<T, Dim + 1> &out_gradient, Optimizer<T> &optimizer, int minibatchInd = 1) override {
            return (input_map[minibatchInd] * (input_map[minibatchInd].constant(1) - input_map[minibatchInd])) * out_gradient;
        }
    };

    template<class T, size_t Dim>
    class Softmax : public Activation<T, Dim> {
    public:
        tbb::concurrent_unordered_map<int, Tensor<T, Dim + 1>> output_map;

        Softmax() : Activation<T, Dim>() {}

        Tensor<T, Dim + 1> forward(const Tensor<T, Dim + 1> &inputs, int minibatchInd = 1, bool train = true) override {
            Tensor<T, Dim + 1> exp_inputs = inputs.exp();

            Tensor<T, Dim + 1> input_tensor_sum = exp_inputs.sum(Eigen::array<long, 1>{1}).broadcast(
                    Eigen::array<size_t, Dim + 1>{1, size_t(inputs.dimension(1)), 1}).reshape(
                    Eigen::array<size_t, Dim + 1>{size_t(inputs.dimension(0)), size_t(inputs.dimension(1)), 1});

            Tensor<T, Dim + 1> tmp_output = (exp_inputs / input_tensor_sum);

            size_t output_size = tmp_output.dimension(1);

            Eigen::array<size_t, 3> new_shape{size_t(tmp_output.dimension(0)), size_t(tmp_output.dimension(1)),
                                              size_t(tmp_output.dimension(1))};

            Eigen::array<size_t, Dim + 1> broadcast_shape{1, 1, output_size};

            Tensor<T, Dim + 1> res = tmp_output.broadcast(broadcast_shape).template reshape(new_shape);

            if (train) {
                auto it = output_map.find(minibatchInd);
                if (it != output_map.end()) {
                    output_map[minibatchInd] = res;
                }
                else {
                    output_map.emplace(minibatchInd, res);
                }
            }
            return tmp_output;
        }

        Tensor<T, Dim + 1>
        backward(const Tensor<T, Dim + 1> &out_gradient, Optimizer<T> &optimizer, int minibatchInd = 1) override {

            Eigen::Tensor<T, Dim + 1> transposed = output_map[minibatchInd].shuffle(Eigen::array<int, 3>{0, 2, 1});

            Eigen::MatrixXd m = Eigen::MatrixXd::Identity(output_map[minibatchInd].dimension(1), output_map[minibatchInd].dimension(1));

            Eigen::TensorMap<const Tensor<T, Dim + 1>> t(m.data(), 1, output_map[minibatchInd].dimension(1),
                                                         output_map[minibatchInd].dimension(1));

            Tensor<T, Dim + 1> identity = t.broadcast(
                    Eigen::array<size_t, 3>({size_t(output_map[minibatchInd].dimension(0)), 1, 1}));

            Eigen::Tensor<T, Dim + 1> result = output_map[minibatchInd] * (identity - transposed);

            Eigen::Tensor<T, Dim + 1> broad_grads = out_gradient.broadcast(
                    Eigen::array<int, 3>{1, 1, int(result.dimension(2))});

            result *= broad_grads.shuffle(Eigen::array<int, 3>{0, 2, 1});

            Tensor<double, 3> res = result.sum(Eigen::array<long, 1>{2}).reshape(
                    Eigen::array<long, 3>{output_map[minibatchInd].dimension(0), output_map[minibatchInd].dimension(2), 1});

            return res;
        }
    };
}


#endif //NEURALIB_ACTIVATION_H