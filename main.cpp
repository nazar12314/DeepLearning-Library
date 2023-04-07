#include <iostream>
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils/MnistDataset.h"
#include "models/Model.h"
#include "utils/Optimizer.h"
#include "utils/Loss.h"
#include "layers/Dense.h"
#include "layers/Activation.h"

int main(int argc, char* argv[]) {

    MnistDataset<double> mnst;
    TensorHolder<double> training_labels = mnst.get_training_labels();
    TensorHolder<double> training_data = mnst.get_training_images();
    initializers::GlorotNormal<double> initializer;
    initializer.set_seed(42);

    Model<double> model("model", new optimizers::SGD<double>(0.001), new loss_functions::BinaryCrossEntropy<double>());
    model.addLayer(new DenseLayer<double>(784, 300, "dense 1", initializer));
    model.addLayer(new activations::ReLU<double>("relu 1"));
    model.addLayer(new DenseLayer<double>(300, 10, "dense 2", initializer));
    model.addLayer(new activations::Softmax<double>("softmax"));
    model.fit(training_data, training_labels, 1);


    return 0;
}