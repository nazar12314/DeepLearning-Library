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

    TensorHolder<T> get_training_labels() {
        Eigen::TensorMap<Eigen::Tensor<uint8_t, 3>> labels(
                dataset.training_labels.data(),
                dataset.training_labels.size(),
                1,
                1
        );

        Tensor<T, 3> labels_tensor = labels.cast<T>();

        return TensorHolder<T>(labels_tensor);
    }

    TensorHolder<T> get_test_labels() {
        Eigen::TensorMap<Eigen::Tensor<uint8_t, 3>> labels(
                dataset.test_labels.data(),
                dataset.test_labels.size(),
                1,
                1
        );

        Tensor<T, 3> labels_tensor = labels.cast<T>();

        return TensorHolder<T>(labels_tensor);
    }

    TensorHolder<T> get_training_images() {
        Eigen::TensorMap<Eigen::Tensor<uint8_t , 3>> images(
                dataset.training_images[0].data(),
                dataset.training_images.size(),
                dataset.training_images[0].size(),
                1
        );

        Tensor<T, 3> images_tensor = images.cast<T>();

        return TensorHolder<T>(images_tensor);
    }

    TensorHolder<T> get_test_images() {
        Eigen::TensorMap<Eigen::Tensor<uint8_t, 3>> images(
                dataset.test_images[0].data(),
                dataset.test_images.size(),
                dataset.test_images[0].size(),
                1
        );

        Tensor<T, 3> images_tensor = images.cast<T>();

        return TensorHolder<T>(images_tensor);
    }
};

#endif //NEURALIB_MNISTDATASET_H
