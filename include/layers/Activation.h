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

    /**
     * Set the weights for the activation function.
     * This method throws an exception as weights are not implemented for Activation class.
     *
     * @param weights_ The weights tensor.
     */
    void set_weights(const Tensor<T, Dim, Eigen::RowMajor> &weights_) {
        throw std::logic_error("Weights aren't implemented for Activation class");
    }

    /**
     * Get the weights for the activation function.
     * This method throws an exception as weights are not implemented for Activation class.
     *
     * @return The weights tensor.
     */
    const Tensor<T, Dim, Eigen::RowMajor> &get_weights() {
        throw std::logic_error("Weights aren't implemented for Activation class");
    }

    ~Activation() = default;
};

namespace activations {

    template<class T, size_t Dim>
    class ReLU : public Activation<T, Dim> {
        tbb::concurrent_unordered_map<int, Tensor<T, Dim+1, Eigen::RowMajor>> input_map;

    public:
        ReLU() : Activation<T, Dim>() {}

        /**
         * Apply the ReLU activation function to the input tensor.
         * If `train` is set to true, it saves the input tensor for the specified minibatch index.
         *
         * @param inputs_ The input tensor.
         * @param minibatchInd The minibatch index.
         * @param train A flag indicating whether training is active or not.
         * @return The tensor after applying the ReLU activation function.
         */
        Tensor<T, Dim+1, Eigen::RowMajor>
        forward(const Tensor<T, Dim+1, Eigen::RowMajor> &inputs_, int minibatchInd = 1, bool train = true) override {

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

        /**
         * Perform the backward pass of the ReLU activation function.
         * It applies the derivative of the ReLU function element-wise to the saved input tensor
         * corresponding to the specified minibatch index and returns the resulting tensor.
         *
         * @param out_gradient The output gradient tensor.
         * @param optimizer The optimizer object.
         * @param minibatchInd The minibatch index.
         * @return The tensor after the backward pass of the ReLU activation function.
         */
        Tensor<T, Dim+1, Eigen::RowMajor>
        backward(const Tensor<T, Dim+1, Eigen::RowMajor> &out_gradient, Optimizer<T> &optimizer, int minibatchInd = 1) override {
            auto relu_derivative = [](T x) { return x > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0); };
            return out_gradient * input_map[minibatchInd].unaryExpr(relu_derivative);
        }

        /**
         * Get the saved input tensor corresponding to the specified minibatch index.
         * If the minibatch index is out of range, an exception is thrown.
         *
         * @param minibatchInd The minibatch index.
         * @return The saved input tensor.
         */
        Tensor<T, Dim+1, Eigen::RowMajor> &get_saved_minibatch(int minibatchInd) override{
            auto it = input_map.find(minibatchInd);
            if (it != input_map.end()) {
                return input_map[minibatchInd];
            }
            else {
                throw std::out_of_range ("Minibatch index is out of range!");
            }
        };
    };

    template<class T, size_t Dim>
    class Sigmoid : public Activation<T, Dim> {
        tbb::concurrent_unordered_map<int, Tensor<T, Dim+1, Eigen::RowMajor>> input_map;
    public:
        Sigmoid() : Activation<T, Dim>() {}

        /**
         * Apply the Sigmoid activation function to the input tensor.
         * If `train` is set to true, it saves the input tensor for the specified minibatch index.
         *
         * @param inputs_ The input tensor.
         * @param minibatchInd The minibatch index.
         * @param train A flag indicating whether training is active or not.
         * @return The tensor after applying the Sigmoid activation function.
         */
        Tensor<T, Dim+1, Eigen::RowMajor>
        forward(const Tensor<T, Dim+1, Eigen::RowMajor> &inputs_, int minibatchInd = 1, bool train = true) override {

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

        /**
         * Perform the backward pass of the Sigmoid activation function.
         * It applies the derivative of the Sigmoid function element-wise to the saved input tensor
         * corresponding to the specified minibatch index and returns the resulting tensor.
         *
         * @param out_gradient The output gradient tensor.
         * @param optimizer The optimizer object.
         * @param minibatchInd The minibatch index.
         * @return The tensor after the backward pass of the Sigmoid activation function.
         */
        Tensor<T, Dim+1, Eigen::RowMajor>
        backward(const Tensor<T, Dim+1, Eigen::RowMajor> &out_gradient, Optimizer<T> &optimizer, int minibatchInd = 1) override {
            return (input_map[minibatchInd] * (input_map[minibatchInd].constant(1) - input_map[minibatchInd])) * out_gradient;
        }

        /**
         * Get the saved input tensor corresponding to the specified minibatch index.
         * If the minibatch index is out of range, an exception is thrown.
         *
         * @param minibatchInd The minibatch index.
         * @return The saved input tensor.
         */
        Tensor<T, Dim+1, Eigen::RowMajor> &get_saved_minibatch(int minibatchInd) override{
            auto it = input_map.find(minibatchInd);
            if (it != input_map.end()) {
                return input_map[minibatchInd];
            }
            else {
                throw std::out_of_range ("Minibatch index is out of range!");
            }
        };
    };

    template<class T, size_t Dim>
    class Softmax : public Activation<T, Dim> {
    public:
        tbb::concurrent_unordered_map<int, Tensor<T, Dim+1, Eigen::RowMajor>> output_map;

        Softmax() : Activation<T, Dim>() {}

        /**
         * Apply the Softmax activation function to the input tensor.
         * If `train` is set to true, it saves the output tensor for the specified minibatch index.
         *
         * @param inputs The input tensor.
         * @param minibatchInd The minibatch index.
         * @param train A flag indicating whether training is active or not.
         * @return The tensor after applying the Softmax activation function.
         */
        Tensor<T, Dim+1, Eigen::RowMajor> forward(const Tensor<T, Dim+1, Eigen::RowMajor> &inputs, int minibatchInd = 1, bool train = true) override {
            Tensor<T, Dim+1, Eigen::RowMajor> exp_inputs = inputs.exp();

            Tensor<T, Dim+1, Eigen::RowMajor> input_tensor_sum = exp_inputs.sum(Eigen::array<long, 1>{1}).broadcast(
                    Eigen::array<size_t, Dim + 1>{1, size_t(inputs.dimension(1)), 1}).reshape(
                    Eigen::array<size_t, Dim + 1>{size_t(inputs.dimension(0)), size_t(inputs.dimension(1)), 1});

            Tensor<T, Dim+1, Eigen::RowMajor> tmp_output = (exp_inputs / input_tensor_sum);

            size_t output_size = tmp_output.dimension(1);

            Eigen::array<size_t, 3> new_shape{size_t(tmp_output.dimension(0)), size_t(tmp_output.dimension(1)),
                                              size_t(tmp_output.dimension(1))};

            Eigen::array<size_t, Dim + 1> broadcast_shape{1, 1, output_size};

            Tensor<T, Dim+1, Eigen::RowMajor> res = tmp_output.broadcast(broadcast_shape).template reshape(new_shape);

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

        /**
         * Perform the backward pass of the Softmax activation function.
         * It computes the gradient with respect to the input tensor using the output gradient tensor,
         * and returns the resulting tensor.
         *
         * @param out_gradient The output gradient tensor.
         * @param optimizer The optimizer object.
         * @param minibatchInd The minibatch index.
         * @return The tensor after the backward pass of the Softmax activation function.
         */
        Tensor<T, Dim+1, Eigen::RowMajor>
        backward(const Tensor<T, Dim+1, Eigen::RowMajor> &out_gradient, Optimizer<T> &optimizer, int minibatchInd = 1) override {

            Eigen::Tensor<T, Dim+1, Eigen::RowMajor> transposed = output_map[minibatchInd].shuffle(Eigen::array<int, 3>{0, 2, 1});

            Eigen::MatrixXd m = Eigen::MatrixXd::Identity(output_map[minibatchInd].dimension(1), output_map[minibatchInd].dimension(1));

            Eigen::TensorMap<const Tensor<T, Dim+1, Eigen::RowMajor>> t(m.data(), 1, output_map[minibatchInd].dimension(1),
                                                         output_map[minibatchInd].dimension(1));

            Tensor<T, Dim+1, Eigen::RowMajor> identity = t.broadcast(
                    Eigen::array<size_t, 3>({size_t(output_map[minibatchInd].dimension(0)), 1, 1}));

            Eigen::Tensor<T, Dim+1, Eigen::RowMajor> result = output_map[minibatchInd] * (identity - transposed);

            Eigen::Tensor<T, Dim+1, Eigen::RowMajor> broad_grads = out_gradient.broadcast(
                    Eigen::array<int, 3>{1, 1, int(result.dimension(2))});

            result *= broad_grads.shuffle(Eigen::array<int, 3>{0, 2, 1});

            Tensor<T, 3, Eigen::RowMajor> res = result.sum(Eigen::array<long, 1>{2}).reshape(
                    Eigen::array<long, 3>{output_map[minibatchInd].dimension(0), output_map[minibatchInd].dimension(2), 1});

            return res;
        }

        /**
         * Get the saved minibatch tensor for the specified minibatch index.
         * This method should not be called from the Softmax activation.
         *
         * @param minibatchInd The minibatch index.
         * @return The saved minibatch tensor.
         * @throws std::out_of_range if this method is called from Softmax.
         */
        Tensor<T, Dim+1, Eigen::RowMajor> &get_saved_minibatch(int minibatchInd) override{
            throw std::out_of_range ("Cant call this method from softmax!");
        };
    };
}


#endif //NEURALIB_ACTIVATION_H