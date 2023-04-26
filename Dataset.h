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

template <class T, size_t DimX, size_t DimY>
class Dataset {
public:
    Tensor<T, DimX> X_train, X_test;
    Tensor<T, DimY> y_train, y_test;

public:

    Dataset() = default;

    Dataset(const std::string& path_to_file) {}

//    bool load_from_file(const std::string& path_to_file) {
//        // Open the CSV file
//        std::ifstream input_file(path_to_file);
//        if (!input_file.is_open()) {
//            std::cerr << "Failed to open file: " << path_to_file << std::endl;
//            return false;
//        }
//
//        // Read the data from the CSV file using Boost tokenizer
//        std::vector<std::vector<T>> data;
//        std::string line;
//        while (std::getline(input_file, line)) {
//            boost::tokenizer<boost::escaped_list_separator<char>> tokenizer(line);
//            std::vector<T> row;
//            for (auto it = tokenizer.begin(); it != tokenizer.end(); ++it) {
//                T value;
//                try {
//                    value = boost::lexical_cast<T>(*it);
//                }
//                catch (const boost::bad_lexical_cast&) {
//                    std::cerr << "Invalid data value: " << *it << std::endl;
//                    return false;
//                }
//                row.push_back(value);
//            }
//            data.push_back(row);
//        }
//
//        // Convert the data to Eigen tensor format
//        size_t num_rows = data.size();
//        size_t num_cols = data[0].size();
//        X_train.resize(num_rows, DimX);
//        y_train.resize(num_rows, DimY);
//        for (size_t i = 0; i < num_rows; ++i) {
//            for (size_t j = 0; j < num_cols - DimY; ++j) {
//                X_train(i, j) = data[i][j];
//            }
//            for (size_t j = num_cols - DimY; j < num_cols; ++j) {
//                y_train(i, j - num_cols + DimY) = data[i][j];
//            }
//        }
//
//        // Close the input file and return success
//        input_file.close();
//        return true;
//    }

    // Split the data in the dataset into training and testing sets
    void split_data(double train_ratio = 0.8) {
        if (train_ratio <= 0.0 || train_ratio >= 1.0) {
            throw std::invalid_argument("Invalid train ratio. It must be in the range (0, 1).");
        }

        // Shuffle the indices of the data points
        std::vector<size_t> indices(X_train.dimension(0));
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine());

        // Compute the number of training and testing data points based on the given ratio
        const auto num_train = static_cast<size_t>(X_train.dimension(0) * train_ratio);
        const size_t num_test = X_train.dimension(0) - num_train;

        // Split the data into training and testing sets
        X_test = X_train.slice(num_train, 0);
        y_test = y_train.slice(num_train, 0);
        X_train = X_train.slice(0, 0, num_train);
        y_train = y_train.slice(0, 0, num_train);
    }
};

#endif //NEURALIB_DATASET_H
