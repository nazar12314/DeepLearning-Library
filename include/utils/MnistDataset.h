//
// Created by Nazar Kononenko on 05.04.2023.
//

#ifndef NEURALIB_MNISTDATASET_H
#define NEURALIB_MNISTDATASET_H

#include "unsupported/Eigen/CXX11/Tensor"
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

using Eigen::Tensor;

template<class T>
class MnistDataset {
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
public:
//    MnistDataset() {
//        mnist::binarize_dataset(dataset);
//    }

    Tensor<T, 3> get_training_labels() {
        Eigen::TensorMap<Eigen::Tensor<uint8_t, 1>> labels(
                dataset.training_labels.data(),
                dataset.training_labels.size()
        );

        Tensor<T, 1> labels_tensor = labels.cast<T>();
        size_t numSamples = labels_tensor.size();

        Tensor<T, 3> oneHotEncoded(numSamples, 10, 1);
        for (int i = 0; i < numSamples; i++) {
            int val = labels_tensor(i);
            oneHotEncoded(i, val-1, 0) = 1;
        }
        return oneHotEncoded;
    }

    Tensor<T, 3> get_test_labels() {
        // TO IMPLEMENT!!!
        Eigen::TensorMap<Eigen::Tensor<uint8_t, 3>> labels(
                dataset.test_labels.data(),
                dataset.test_labels.size(),
                1,
                1
        );

        Tensor<T, 3> labels_tensor = labels.cast<T>();
        return labels_tensor;
    }

    Tensor<T, 3> get_training_images() {
        Eigen::TensorMap<Eigen::Tensor<uint8_t , 3>> images(
                dataset.training_images[0].data(),
                dataset.training_images.size(),
                dataset.training_images[0].size(),
                1
        );
        Tensor<T, 3> images_tensor = images.cast<T>();
        images_tensor /= images_tensor.constant(255.0);
        return images_tensor;
    }

    Tensor<T, 3> get_test_images() {
        // TO IMPLEMENT!!!
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
