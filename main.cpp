#include <iostream>
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils/MnistDataset.h"

int main(int argc, char* argv[]) {
    MnistDataset<double> mnst;

    TensorHolder<double> training_images = mnst.get_training_images();

    std::cout << training_images.get<2>().dimensions() << std::endl;

    return 0;
}