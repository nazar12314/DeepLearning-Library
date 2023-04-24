#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "models/Model.h"
#include "utils/Initializer.h"
#include "layers/Dense.h"
#include "utils/Optimizer.h"
#include "layers/Activation.h"

int main() {


    Tensor<double, 3> data(10, 3, 1);
    data.setConstant(1);

    initializers::GlorotNormal<double> initializer;
    initializer.set_seed(42);


    Model<double, 3, 3> model("model", new optimizers::SGD<double>(0.001), new loss_functions::MSE<double>());

    DenseLayer<double> layer (3, 5, "dense 1", initializer);
    DenseLayer<double> layer2 (5, 10, "dense 2", initializer);
    DenseLayer<double> layer3 (10, 3, "dense 2", initializer);
    auto input = model.addLayer(layer);
    auto hidden = model.addLayer(layer2);
    auto hidden2 = model.addLayer(layer3);
//    auto activatoin = model.addLayer(relu);
    connectLayers(input, hidden);
    connectLayers(hidden, hidden2);
//    connectLayers(hidden2, activatoin);

    model.setInput(input);
    model.setOut(hidden2);

    model.fit(data, data, 10);


    return 0;
}
