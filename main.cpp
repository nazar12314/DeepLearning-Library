#include <iostream>
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils/MnistDataset.h"

int main(int argc, char* argv[]) {
    MnistDataset mnst;

    TensorHolder<uint8_t> training_labels = mnst.get_training_labels();

    std::cout << training_labels.get<1>() << std::endl;

    return 0;
}