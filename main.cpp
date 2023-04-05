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
    initializers::RandomNormal<double> initializer(0);

    Tensor<double, 3> X_train(4, 2, 1);
    // 4 instances
    // each instance is 2 by 1
    X_train(0, 0, 0) = 0;
    X_train(0, 1, 0) = 0;

    X_train(1, 0, 0) = 0;
    X_train(1, 1, 0) = 1;

    X_train(2, 0, 0) = 1;
    X_train(2, 1, 0) = 0;

    X_train(3, 0, 0) = 1;
    X_train(3, 1, 0) = 1;

    Tensor<double, 3> y_train(4, 1, 1);
    y_train(0, 0, 0) = 0;
    y_train(1, 0, 0) = 1;
    y_train(2, 0, 0) = 1;
    y_train(3, 0, 0) = 0;


    Model<double> model("model", new optimizers::SGD<double>(0.001), new loss_functions::BinaryCrossEntropy<double>());
    model.addLayer(new DenseLayer<double>(2, 10, "dense 1", initializer));
    model.addLayer(new activations::ReLU<double>("relu"));
    model.addLayer(new DenseLayer<double>(10, 20, "dense 2", initializer));
    model.addLayer(new activations::ReLU<double>("relu2"));
    model.addLayer(new DenseLayer<double>(20, 1, "dense 3", initializer));
    model.fit(TensorHolder<double>(X_train), TensorHolder<double>(y_train), 50);

    std::cout << "End of training.\nPredictions:\n";

    for (int i=0; i < X_train.dimension(0); ++i){
        TensorHolder<double> X_test {Tensor<double, 2>{X_train.chip(i, 0)}};
        TensorHolder<double> y_true {Tensor<double, 2>{y_train.chip(i, 0)}};
        std::cout << "X: " << X_test.get<2>()(0, 0) <<" "<< X_test.get<2>()(1, 0) << std::endl;
        std::cout << "Y_true: " << y_train.chip(i, 0) << std::endl;
        std::cout << "Y_pred: " << model.predict(X_test).get<2>() << std::endl << std::endl;
    }
    return 0;
}