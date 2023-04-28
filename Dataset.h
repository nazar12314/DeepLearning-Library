//
// Created by nazar on 24.04.23.
//

#ifndef NEURALIB_DATASET_H
#define NEURALIB_DATASET_H

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>
#include <boost/tokenizer.hpp>
#include "eigen3/Eigen/Core"

using Eigen::Tensor;

template <typename T>
std::vector<T> fileToVector(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::logic_error("Cannot open file of labels");
    }
    std::vector<T> values;
    // Read the file contents into a string
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Parse the string to obtain values of type T separated by spaces
    std::stringstream stream(content);
    T value;
    while (stream >> value) {
        values.push_back(value);
    }
    return values;
}

template <class T, size_t Dim>
void vectorToVectorOfTensors(std::vector<Eigen::Tensor<T, Dim>>& result, std::vector<T>& data, size_t rows, size_t cols, size_t depth = 0) {

    size_t numTensors = data.size() / (rows * cols);

    if constexpr (Dim == 2) {
        for (size_t i = 0; i < numTensors; ++i) {
            Eigen::TensorMap<Eigen::Tensor<T, 2>> tensor(data.data() + i * rows * cols, rows, cols);
            result.push_back(tensor);
        }
    } else {
        for (size_t i = 0; i < numTensors; ++i) {
            Eigen::TensorMap<Eigen::Tensor<T, 3>> tensor(data.data() + i * depth * rows * cols, depth, rows, cols);
            result.push_back(tensor);
        }
    }
}

template<class T, size_t NumDimensions>
Eigen::Tensor<T, NumDimensions> vectorOfTensorsToTensor(const std::vector<Eigen::Tensor<T, NumDimensions>>& tensors) {
    // Determine the size of the resulting tensor
    int size = tensors.size() * tensors[0].size();

    // Create a new tensor with the appropriate size
    Eigen::Tensor<T, NumDimensions> result(tensors[0].dimensions());
    result.resize(size, Eigen::NoChange);

    // Concatenate the data from all the tensors into the new tensor
    for (int i = 0; i < tensors.size(); ++i) {
        Eigen::array<int, NumDimensions> offsets;
        offsets.fill(i * tensors[i].size());
        result.slice(offsets, tensors[i].dimensions()) = tensors[i];
    }

    return result;
}



template<class TX, class TY, size_t DimX, size_t DimY>
class Dataset {
public:
    std::vector<Tensor<TX, DimX>> X_train, X_test;
    std::vector<Tensor<TY, DimY>> y_train, y_test;

    std::vector<size_t> dimsX, dimsy;

public:
    Dataset(const std::vector<size_t> &dimsX, const std::vector<size_t> &dimsy) : dimsX(dimsX), dimsy(dimsy) {
        this->dimsX.resize(DimX, 0);
        this->dimsX.resize(DimY, 0);
    };

    void read_from_files(const std::string &path_to_labels, const std::string &path_to_targets) {

        auto labels = fileToVector<TX>(path_to_labels);
        if constexpr (DimX == 2) {
            vectorToVectorOfTensors<TX, 2>(X_train,labels, dimsX[0], dimsX[1], 0);
        } else {
             vectorToVectorOfTensors<TX, 3>(X_train, labels, dimsX[0], dimsX[1], dimsX[2]);
        }

        auto targets = fileToVector<TX>(path_to_targets);
        if constexpr (DimY == 2) {
            vectorToVectorOfTensors<TY, 2>(y_train,targets, dimsX[0], dimsX[1], 0);
        } else {
            vectorToVectorOfTensors<TY, 3>(y_train, targets, dimsX[0], dimsX[1], dimsX[2]);
        }

    }

    // Splits the data in the dataset into training and testing sets
    void split_data(double train_ratio = 0.8) {
        // Step 1: Calculate the number of samples to put in the test set.
        size_t num_samples = X_train.size();
        size_t num_test_samples = static_cast<size_t>(num_samples * (1.0 - train_ratio));

        // Step 2: Randomly shuffle the indices of the samples.
        std::vector<size_t> indices(num_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));

        // Step 3: Split the shuffled indices into training and test indices.
        std::vector<size_t> train_indices(indices.begin(), indices.begin() + (num_samples - num_test_samples));
        std::vector<size_t> test_indices(indices.begin() + (num_samples - num_test_samples), indices.end());

        // Step 4: Create X_test and y_test vectors by copying the corresponding samples from X_train and y_train using the test indices.
        X_test.clear();
        y_test.clear();
        for (const auto& index : test_indices) {
            X_test.push_back(X_train[index]);
            y_test.push_back(y_train[index]);
        }

        // Step 5: Remove the test samples from X_train and y_train using the training indices.
        std::vector<Tensor<TX, DimX>> new_X_train;
        std::vector<Tensor<TY, DimY>> new_y_train;
        for (const auto& index : train_indices) {
            new_X_train.push_back(X_train[index]);
            new_y_train.push_back(y_train[index]);
        }
        X_train = new_X_train;
        y_train = new_y_train;
    }

};

#endif //NEURALIB_DATASET_H
