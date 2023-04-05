//
// Created by Nazar Kononenko on 05.04.2023.
//

#ifndef NEURALIB_MNISTDATASET_H
#define NEURALIB_MNISTDATASET_H

#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils/TensorHolder.h"
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

using Eigen::Tensor;

template<class T>
class MnistDataset {
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
public:
    MnistDataset() {
        mnist::binarize_dataset(dataset);
    }

    TensorHolder<T> get_training_images() {
        const size_t num_images = dataset.training_images.size();
        const size_t image_size = dataset.training_images[0].size();

        Eigen::TensorMap<Eigen::Tensor<uint8_t , 2>> images(
                dataset.training_images[0].data(),
                num_images,
                image_size
                );

        Tensor<T, 2> images_tensor = images.cast<T>();

        return TensorHolder<T>(images_tensor);
    }

    TensorHolder<uint8_t> get_training_labels() {
        Eigen::TensorMap<Eigen::Tensor<uint8_t, 1>> labels(
                dataset.training_labels.data(),
                dataset.training_labels.size()
                );

        constexpr size_t data_size = 60000;

        Eigen::Tensor<uint8_t, 2> tensor_labels = labels.reshape(
                Eigen::array<Eigen::Index,
                2>({data_size, 1})
                );

        return TensorHolder<uint8_t>(tensor_labels);
    }

    TensorHolder<T> get_test_images() {
        const size_t num_images = dataset.test_images.size();
        const size_t image_size = dataset.test_images[0].size();

        Eigen::TensorMap<Eigen::Tensor<uint8_t, 2>> images(
                dataset.test_images[0].data(),
                num_images,
                image_size
        );

        Tensor<T, 2> images_tensor = images.cast<T>();

        return TensorHolder<T>(images_tensor);
    }

    TensorHolder<uint8_t> get_test_labels() {
        Eigen::TensorMap<Eigen::Tensor<uint8_t, 1>> labels(
                dataset.test_labels.data(),
                dataset.test_labels.size()
        );

        constexpr size_t data_size = 10000;

        Eigen::Tensor<uint8_t, 2> tensor_labels = labels.reshape(
                Eigen::array<Eigen::Index,
                        2>({data_size, 1})
        );

        return TensorHolder<uint8_t>(tensor_labels);
    }
};

#endif //NEURALIB_MNISTDATASET_H
