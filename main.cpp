#include <iostream>
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils/MnistDataset.h"

int main(int argc, char* argv[]) {
    MnistDataset<double> mnst;

    TensorHolder<uint8_t> training_labels = mnst.get_test_labels();

    std::cout << training_labels.get<2>().dimensions() << std::endl;

    return 0;
}