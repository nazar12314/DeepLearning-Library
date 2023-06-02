//
// Created by Nazar Kononenko on 31.05.2023.
//

#ifndef NEURALIB_CONVOLUTION_H
#define NEURALIB_CONVOLUTION_H

#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/Initializer.h"
#include "Layer.h"
#include "utils/Optimizer.h"
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>
#include <Eigen/Core>
#include <map>

using Eigen::Tensor;

/**
 * ConvolutionLayer Class
 *
 * This class represents a convolutional layer in a neural network.
 * It performs convolutional operations on input data using a set of kernels and biases.
 *
 * Template Parameters:
 *   - T: The data type used for the computations (e.g., float, double)
 *   - Dim: The dimensionality of the input data (default: 3)
 *
 * Public Methods:
 *   - ConvolutionLayer(...): Constructor for creating a ConvolutionLayer object.
 *   - forward(...): Performs forward pass computation of the convolutional layer.
 *   - backward(...): Performs backward pass computation of the convolutional layer.
 *   - get_saved_minibatch(...): Retrieves the saved minibatch for a specific index.
 *   - get_kernels(): Retrieves the convolutional kernels used in the layer.
 */

template<class T, size_t Dim = 3>
class ConvolutionLayer : public Layer<T, Dim> {
    size_t input_height;
    size_t input_width;
    size_t input_depth;
    long kernel_size;
    size_t kernel_depth;
    std::mutex mutex;

    Tensor<T, Dim+1, Eigen::RowMajor> kernels;
    Tensor<T, Dim, Eigen::RowMajor> biases;

    tbb::concurrent_unordered_map<int, Tensor<T, Dim+1, Eigen::RowMajor>> input_hash_map;
public:
    /**
     * Constructor
     *
     * Creates a ConvolutionLayer object with the specified parameters.
     *
     * @param input_height_   Height of the input data
     * @param input_width_    Width of the input data
     * @param input_depth_    Depth of the input data
     * @param kernel_size_    Size of the convolutional kernels
     * @param kernel_depth_   Depth of the convolutional kernels
     * @param name            Name of the layer
     * @param initializer     Object for initializing the layer's weights and biases
     * @param trainable       Flag indicating whether the layer is trainable (default: true)
   */
    ConvolutionLayer
    (size_t input_height_, size_t input_width_, size_t input_depth_, size_t kernel_size_, size_t kernel_depth_, const std::string& name, Initializer<T>& initializer, bool trainable = true)
    :
    Layer<T, Dim>(name, trainable),
    input_height(input_height_),
    input_width(input_width_),
    input_depth(input_depth_),
    kernel_size(kernel_size_),
    kernel_depth(kernel_depth_),
    kernels{initializer.get_weights_4d(kernel_size, input_depth, kernel_depth)},
    biases{initializer.get_biases_3d(input_height - kernel_size + 1, input_width - kernel_size + 1, kernel_depth)}
    {}

    /**
     * forward
     *
     * Performs the forward pass computation of the convolutional layer.
     *
     * @param inputs          Input data tensor
     * @param minibatchInd    Index of the current minibatch
     * @param train           Flag indicating if the layer is in training mode (default: true)
     * @return                Result of the forward pass computation
     */
    Tensor<T, Dim+1, Eigen::RowMajor> forward(const Tensor<T, Dim+1, Eigen::RowMajor> &inputs, int minibatchInd = 1, bool train = true) override {
        if (train) {
            auto it = input_hash_map.find(minibatchInd);
            if (it != input_hash_map.end()) {
                input_hash_map[minibatchInd] = inputs;
            }
            else {
                input_hash_map.emplace(minibatchInd, inputs);
            }
        }
        Tensor<double, 4, Eigen::RowMajor> result (inputs.dimension(0), input_height - kernel_size + 1, input_width - kernel_size + 1, kernel_depth);
        result.setConstant(0);
        Eigen::array<ptrdiff_t, 2> dims({1, 2});

        for (int i = 0; i < kernel_depth; i++) {
            result.chip(i, 3) += inputs.convolve(kernels.chip(i, 3), dims);
        }
        for (int i = 0; i < inputs.dimension(0); ++i) {
            result.chip(i, 0) += biases;
        }
        return result;
    }

    /**
     * backward
     *
     * Performs the backward pass computation of the convolutional layer.
     *
     * @param out_gradient    Gradient tensor from the subsequent layer
     * @param optimizer       Optimizer object for updating the layer's parameters
     * @param minibatchInd    Index of the current minibatch
     * @return                Gradient tensor to be propagated back to the previous layer
     */
    Tensor<T, Dim+1, Eigen::RowMajor> backward(const Tensor<T, Dim+1, Eigen::RowMajor> &out_gradient, Optimizer<T> &optimizer, int minibatchInd = 1) override {
        Tensor<T, Dim+1, Eigen::RowMajor> kernels_gradients = kernels.constant(0);
        Tensor<T, Dim+1, Eigen::RowMajor> inputs_minibatch = input_hash_map[minibatchInd];
        Tensor<T, Dim+1, Eigen::RowMajor> input_gradients = Tensor<T, Dim+1, Eigen::RowMajor>(
                inputs_minibatch.dimension(0), inputs_minibatch.dimension(1), inputs_minibatch.dimension(2), inputs_minibatch.dimension(3));
        input_gradients.setConstant(0.0);

        Eigen::array<ptrdiff_t, 2> dims({1, 2});
        for (int batch_ind = 0; batch_ind < inputs_minibatch.dimension(0); batch_ind++){
            for (int i = 0; i < kernel_depth; ++i) {
                for (int j = 0; j < input_depth; ++j) {
                    kernels_gradients.chip(i, 3).chip(j, 2) += (Tensor<T, 2, Eigen::RowMajor>)(inputs_minibatch
                            .chip(batch_ind, 0).chip(j, 2))
                            .convolve((Tensor<T, 2, Eigen::RowMajor>)(out_gradient.chip(batch_ind, 0).chip(i, 2)), Eigen::array<ptrdiff_t, 2>({0, 1}));
                }
            }
        }
        Eigen::array<std::pair<int, int>, 4> padding;
        padding[0] = std::make_pair(0, 0);
        padding[1] = std::make_pair(kernel_size-1, kernel_size-1);
        padding[2] = std::make_pair(kernel_size-1, kernel_size-1);
        padding[3] = std::make_pair(0, 0);
        Eigen::Tensor<T, 4, Eigen::RowMajor> paddedInput = out_gradient.pad(padding);
        for (int i = 0; i < kernel_depth; ++i) {
            for (int j = 0; j < input_depth; ++j){
                input_gradients.chip(j, 3) += (Tensor<T, 3, Eigen::RowMajor>)(
                        paddedInput.slice(Eigen::array<Eigen::Index, 4>({0, 0, 0, i}),
                                Eigen::array<Eigen::Index, 4>({paddedInput.dimension(0), paddedInput.dimension(1), paddedInput.dimension(2), 1})))
                        .convolve((Tensor<T, 2, Eigen::RowMajor>)(kernels.chip(i, 3).chip(j, 2)).reverse(Eigen::array<int, 2>{0, 1}), dims).chip(0, 3);
            }
//            input_gradients.chip(i, 3) += paddedInput.convolve(kernels.chip(i, 3).reverse(Eigen::array<int, 3>{0, 1, 2}), dims);
        }
        Tensor<T, Dim, Eigen::RowMajor> tmp_biases = out_gradient.sum(Eigen::array<int, 1>{0});
        mutex.lock();
        kernels -= optimizer.apply_optimization4d(kernels_gradients);
        biases -= optimizer.apply_optimization3d(tmp_biases);
        mutex.unlock();
        return input_gradients;
    }

    /**
     * get_saved_minibatch
     *
     * Retrieves the saved minibatch for the specified index.
     *
     * @param minibatchInd    Index of the minibatch to retrieve
     * @return                Saved minibatch tensor
     * @throw std::out_of_range if the minibatch index is out of range
     */
    Tensor<T, Dim+1, Eigen::RowMajor> &get_saved_minibatch(int minibatchInd) override{
        auto it = input_hash_map.find(minibatchInd);
        if (it != input_hash_map.end()) {
            return input_hash_map[minibatchInd];
        }
        else {
            throw std::out_of_range ("Minibatch index is out of range!");
        }
    };

    /**
     * get_kernels
     *
     * Retrieves the convolutional kernels used in the layer.
     *
     * @return    Convolutional kernels tensor
     */
    Tensor<T, Dim+1, Eigen::RowMajor> get_kernels() {
        return kernels;
    }
};

/**
 * MaxPooling Class
 *
 * This class represents a max pooling layer in a neural network.
 * It performs downsampling by selecting the maximum value within each pooling region.
 *
 * Template Parameters:
 *   - T: The data type used for the computations (e.g., float, double)
 *   - Dim: The dimensionality of the input data (default: 3)
 *
 * Public Methods:
 *   - MaxPooling(...): Constructor for creating a MaxPooling object.
 *   - forward(...): Performs forward pass computation of the max pooling layer.
 *   - backward(...): Performs backward pass computation of the max pooling layer.
 *   - get_saved_minibatch(...): Retrieves the saved minibatch for a specific index.
 */
template<class T, size_t Dim = 3>
class MaxPooling: public Layer<T, Dim>{
    int poolSize;
    tbb::concurrent_unordered_map<int, Tensor<T, Dim+1, Eigen::RowMajor>> input_hash_map;
public:
    /**
     * Constructor
     *
     * Creates a MaxPooling object with the specified parameters.
     *
     * @param poolSize_    Size of the pooling regions
     */
    explicit MaxPooling(int poolSize_): Layer<T, Dim>("maxpool", false), poolSize{poolSize_}{}

    /**
     * forward
     *
     * Performs the forward pass computation of the max pooling layer.
     *
     * @param inputs          Input data tensor
     * @param minibatchInd    Index of the current minibatch
     * @param train           Flag indicating if the layer is in training mode (default: true)
     * @return                Result of the forward pass computation
     */
    Tensor<T, Dim+1, Eigen::RowMajor> forward(const Tensor<T, Dim+1, Eigen::RowMajor> &inputs, int minibatchInd = 1, bool train = true) override{
        int batchSize = inputs.dimension(0);
        int inputHeight = inputs.dimension(1);
        int inputWidth = inputs.dimension(2);
        int numChannels = inputs.dimension(3);

        int outputHeight = inputHeight / poolSize;
        int outputWidth = inputWidth / poolSize;

        Tensor<T, Dim+1, Eigen::RowMajor> output(batchSize, outputHeight, outputWidth, numChannels);
        Tensor<T, Dim+1, Eigen::RowMajor> indexes(batchSize, inputHeight, inputWidth, numChannels);
        indexes.setConstant(0);
        Tensor<Eigen::Index, 0, Eigen::RowMajor> max_arg;

        for (int batch_ind = 0; batch_ind < batchSize; ++batch_ind){
            for (int i = 0; i < outputHeight; ++i) {
                for (int j = 0; j < outputWidth; ++j) {
                    for (int c = 0; c < numChannels; ++c) {
                        auto slice = (Tensor<T, Dim+1, Eigen::RowMajor>)inputs.slice(
                                Eigen::array<Eigen::Index, 4>({batch_ind, i * poolSize, j * poolSize, c}),
                                Eigen::array<Eigen::Index, 4>({1, poolSize, poolSize, 1})
                        );
                        max_arg = slice.argmax();
                        int rowIdx = i*poolSize + max_arg(0) / poolSize;
                        int colIdx = j * poolSize + max_arg(0) % poolSize;

                        output(batch_ind, i, j, c) = inputs(batch_ind, rowIdx, colIdx, c);
                        indexes(batch_ind, rowIdx, colIdx, c) = 1;
                    }
                }
            }
        }

        if (train) {
            auto it = input_hash_map.find(minibatchInd);
            if (it != input_hash_map.end()) {
                input_hash_map[minibatchInd] = indexes;
            }
            else {
                input_hash_map.emplace(minibatchInd, indexes);
            }
        }
        return output;
    }

    /**
     * backward
     *
     * Performs the backward pass computation of the max pooling layer.
     *
     * @param out_gradient    Gradient tensor from the subsequent layer
     * @param optimizer       Optimizer object for updating the layer's parameters
     * @param minibatchInd    Index of the current minibatch
     * @return                Gradient tensor to be propagated back to the previous layer
     */
    Tensor<T, Dim + 1, Eigen::RowMajor>
    backward(const Tensor<T, Dim + 1, Eigen::RowMajor> &out_gradient, Optimizer<T> &optimizer, int minibatchInd) override {
        return out_gradient * input_hash_map[minibatchInd];
    }

    /**
     * get_saved_minibatch
     *
     * Retrieves the saved minibatch for the specified index.
     *
     * @param minibatchInd    Index of the minibatch to retrieve
     * @return                Saved minibatch tensor
     * @throw std::out_of_range if the minibatch index is out of range
     */
    Tensor<T, Dim + 1, Eigen::RowMajor> &get_saved_minibatch(int minibatchInd) override {
        auto it = input_hash_map.find(minibatchInd);
        if (it != input_hash_map.end()) {
            return input_hash_map[minibatchInd];
        }
        else {
            throw std::out_of_range ("Minibatch index is out of range!");
        }
    }
};

#endif //NEURALIB_CONVOLUTION_H
