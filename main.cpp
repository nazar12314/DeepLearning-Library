#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include "models/Model.h"
#include "utils/Initializer.h"
#include "layers/Dense.h"
#include "utils/Optimizer.h"
#include "layers/Activation.h"
#include <layers/Convolution.h>
#include "utils/Dataset.h"

#include <fstream>
#include <sstream>
#include <map>


int main() {
    std::string X_path = "../mnist/mnist_train_data.csv";
    std::string y_path = "../mnist/mnist_train_labels.csv";

    auto X_train = read_csv3d<60000, 28, 28, 1>(X_path);
    auto y_train = read_csv2d<60000, 10, 1>(y_path);

    X_train /= X_train.constant(255);

    initializers::GlorotNormal<double> initializer;
    initializer.set_seed(42);

    Model<double, 4, 3> model("model", new optimizers::SGD<double>(0.05), new loss_functions::BinaryCrossEntropy<double>());

    auto input = model.addLayer<ConvolutionLayer<double>, 3>(28, 28, 1, 3, 2, "conv1", initializer);
    auto conv1 = model.addLayer<ConvolutionLayer<double>, 3>(26, 26, 2, 3, 2, "conv2", initializer);
    auto conv2 = model.addLayer<ConvolutionLayer<double>, 3>(24, 24, 2, 3, 2, "conv3", initializer);
    auto conv3 = model.addLayer<ConvolutionLayer<double>, 3>(22, 22, 2, 3, 2, "conv4", initializer);
    auto conv4 = model.addLayer<ConvolutionLayer<double>, 3>(20, 20, 2, 3, 1, "conv5", initializer);
    auto flatten = model.addFlattenLayer();
    auto dense1 = model.addLayer<DenseLayer<double>>(324, 20, "dense1", initializer);
    auto dense2 = model.addLayer<DenseLayer<double>>(20, 10, "dense2", initializer);

    auto sigmoid1 = model.addLayer<activations::Sigmoid<double, 3>, 3>();
    auto sigmoid2 = model.addLayer<activations::Sigmoid<double, 3>, 3>();
    auto sigmoid3 = model.addLayer<activations::Sigmoid<double, 3>, 3>();
    auto sigmoid4 = model.addLayer<activations::Sigmoid<double, 3>, 3>();
    auto sigmoid5 = model.addLayer<activations::Sigmoid<double, 2>>();
    auto sigmoid6 = model.addLayer<activations::Sigmoid<double, 2>>();
    auto out = model.addLayer<activations::Softmax<double, 2>>();

    connect(input, sigmoid1);
    connect(sigmoid1, conv1);
    connect(conv1, sigmoid2);
    connect(sigmoid2, conv2);
    connect(conv2, sigmoid3);
    connect(sigmoid3, conv3);
    connect(conv3, sigmoid4);
    connect(sigmoid4, conv4);
    connect(conv4, flatten);
    connect(flatten, dense1);
    connect(dense1, sigmoid5);
    connect(sigmoid5, dense2);
    connect(dense2, out);

    model.setInput(input);
    model.setOut(out);

    std::cout << "Start:" << std::endl;
    model.fit(X_train, y_train, 10, 200, 4);

    return 0;
}