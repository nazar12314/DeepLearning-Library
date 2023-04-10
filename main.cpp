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

    Tensor<double, 2> ts(3, 3);

    ts.setValues({
                         {1, 2, 3},
                         {3, 3, 3},
                         {1, 4, 2}
    });

    initializers::GlorotNormal<double> initializer;
    initializer.set_seed(42);

    Model<double> model("model", new optimizers::SGD<double>(0.001), new loss_functions::MSE<double>());
    model.addLayer(new DenseLayer<double>(784, 20, "dense 1", initializer));
    model.addLayer(new activations::ReLU<double, 2>());
    model.addLayer(new DenseLayer<double>(20, 10, "dense 2", initializer));
    model.addLayer(new activations::ReLU<double, 2>());

    return 0;
}
