#include <iostream>
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"
#include "eigen3/unsupported/Eigen/CXX11/Tensor"

int main(int argc, char* argv[]) {
    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    mnist::binarize_dataset(dataset);

    Eigen::TensorMap<Eigen::Tensor<uint8_t, 1>> labels(dataset.training_labels.data(), dataset.training_labels.size());

    // Get dimensions of the images
    const size_t num_images = dataset.training_images.size();
    const size_t image_size = dataset.training_images[0].size();

    // Create 3D tensor to hold all training images
    Eigen::TensorMap<Eigen::Tensor<uint8_t, 2>> images(dataset.training_images[0].data(), num_images, image_size);

    std::cout << images.dimensions() << std::endl;

    return 0;
}