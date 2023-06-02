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

std::string read_csv(const std::ifstream &file, size_t &count) {
    // Reading the csv content into a string
    std::string content;

    {
        std::stringstream buffer;
        buffer << file.rdbuf();
        content = buffer.str();
    }

    const std::string s = ",";
    const std::string t = " ";

    std::string::size_type n = 0;
    while ((n = content.find(s, n)) != std::string::npos) {
        content.replace(n, s.size(), t);
        n += t.size();
        count++;
    }

    return content;
}


template<class T, size_t Dim>
void streamToTensor(std::stringstream &stream, Eigen::Tensor<T, Dim + 1> &result, const std::vector<size_t> &dimensions,
                    size_t numTensors) {

    std::array<Eigen::Index, Dim + 1> sizes;
    std::copy(dimensions.begin(), dimensions.end(), sizes.begin() + 1);
    sizes[0] = numTensors;
    result.resize(sizes);

    if constexpr (Dim == 0) {
        for (size_t i = 0; i < numTensors; ++i) {
            stream >> result(i);
        }
    } else if constexpr (Dim == 1) {
        for (size_t i = 0; i < numTensors; ++i) {
            for (size_t col = 0; col < dimensions[0]; ++col) {
                stream >> result(i, col);
            }
        }
    } else if constexpr (Dim == 2) {
        for (size_t i = 0; i < numTensors; ++i) {
            for (size_t row = 0; row < dimensions[0]; ++row) {
                for (size_t col = 0; col < dimensions[1]; ++col) {
                    stream >> result(i, row, col);
                }
            }
        }
    } else {
        for (size_t i = 0; i < numTensors; ++i) {
            for (size_t depth = 0; depth < dimensions[0]; ++depth) {
                for (size_t row = 0; row < dimensions[1]; ++row) {
                    for (size_t col = 0; col < dimensions[2]; ++col) {
                        T value;
                        stream >> value;
                        result(i, depth, row, col) = value;
                    }
                }
            }
        }
    }
}

template<typename T, size_t Dim>
void readFileToTensor(const std::string &filename, Tensor<T, Dim + 1> &tensor, const std::vector<size_t> &dimensions) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::logic_error("Cannot open file of labels");
    }

    // Reading a csv and obtaining the number of entries
    size_t count = 0;
    std::stringstream stream(read_csv(file, count));

    if (!dimensions.empty())
        count = count / std::accumulate(dimensions.begin(), dimensions.end(), (size_t) 1, std::multiplies<T>());


    streamToTensor<T, Dim>(stream, tensor, dimensions, count);
}


template<class TX, class TY, size_t DimX, size_t DimY>
class Dataset {
public:
    Tensor<TX, DimX + 1> X_train, X_test;
    Tensor<TY, DimY + 1> y_train, y_test;

    std::vector<size_t> dimsX, dimsy;

public:
    Dataset(const std::vector<size_t> &dimsX, const std::vector<size_t> &dimsy) : dimsX(dimsX), dimsy(dimsy) {
        this->dimsX.resize(DimX, 0);
        this->dimsX.resize(DimY, 0);
    };

    void read_from_files(const std::string &path_to_labels, const std::string &path_to_targets) {
        readFileToTensor<TX, DimX>(path_to_labels, X_train, dimsX);
        readFileToTensor<TY, DimY>(path_to_targets, y_train, dimsy);
    }
};

template <size_t ROWS, size_t HEIGHT, size_t WEIGHT, size_t CHNALLES>
Tensor<double, 4, Eigen::RowMajor> read_csv3d(const std::string& filename){
    Tensor<double, 4, Eigen::RowMajor> data(ROWS, HEIGHT, WEIGHT, CHNALLES);

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::logic_error("Cannot open file of labels");
    }
    std::string line;
    int row = 0;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        int height = 0;
        int width = 0;
        while (getline(ss, cell, ',')) {
            float value = stof(cell);
            data(row, height, width, 0) = value;
            width++;
            if (width==HEIGHT){
                width = 0;
                height++;
            }
        }
        row++;
    }

    return data;
}

template <size_t ROWS, size_t COLS, size_t CHNALLES>
Tensor<double, 3, Eigen::RowMajor> read_csv2d(const std::string& filename){
    Tensor<double, 3, Eigen::RowMajor> data(ROWS, COLS, CHNALLES);

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::logic_error("Cannot open file of labels");
    }
    std::string line;
    int row = 0;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        int col = 0;
        while (getline(ss, cell, ',')) {
            float value = stof(cell);
            data(row, col, 0) = value;
            col++;
        }
        row++;
    }

    return data;
}
#endif //NEURALIB_DATASET_H
