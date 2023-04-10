#include <iostream>
#include <chrono>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "models/Model.h"
#include "utils/Initializer.h"
#include "layers/Dense.h"
#include "layers/Activation.h"
#include "utils/Optimizer.h"

int main() {
//     Define the matrix and tensor sizes

    Tensor<double, 2> ts(3, 1);
    ts.setConstant(1);

    initializers::GlorotNormal<double> initializer;
    initializer.set_seed(42);

    Model<double> model("model", new optimizers::SGD<double>(0.001), new loss_functions::MSE<double>());
    model.addLayer(new DenseLayer<double>(3, 5, "dense 1", initializer));
    model.addLayer(new activations::ReLU<double, 2>());
    model.addLayer(new DenseLayer<double>(5, 10, "dense 2", initializer));
    model.addLayer(new activations::ReLU<double, 2>());

    std::cout << std::get<Tensor<double, 2>> (model.predict(ts));

    return 0;
}
