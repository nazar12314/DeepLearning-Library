#include <iostream>
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils/MnistDataset.h"

int main(int argc, char* argv[]) {
    MnistDataset<double> mnst;

    TensorHolder<double> training_labels = mnst.get_training_labels();
    TensorHolder<double> test_labels = mnst.get_test_labels();
    TensorHolder<double> training_images = mnst.get_training_images();
    TensorHolder<double> test_images = mnst.get_test_images();

    std::cout << training_images.get<3>().dimensions() << std::endl;
    std::cout << test_images.get<3>().dimensions() << std::endl;
    std::cout << test_labels.get<3>().dimensions() << std::endl;
    std::cout << training_labels.get<3>().dimensions() << std::endl;

    return 0;
}