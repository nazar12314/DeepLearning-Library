#include <iostream>
#include "utils/TensorHolder.h"
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "layers/Dense.h"
#include "utils/Optimizer.h"
#include <chrono>
#include "layers/Dense.h"
#include "utils/Initializer.h"
#include "models/Model.h"
#include "layers/Activation.h"

using namespace Eigen;

int main() {
    initializers::RandomNormal<double> ci (10);
    Model<double> ml;

    Tensor<double, 2> ts (10, 1);
    ts.setRandom();

    TensorHolder<double> th (ts);

    ml.addLayer(new DenseLayer<double> (10, 5, "Dense 1", ci));
    ml.addLayer(new DenseLayer<double> (5, 15, "Dense 2", ci));

    std::cout << ml.predict(th).get<2>() << std::endl;
    Activation<double>* activation = new activations::ReLU<double>();
    auto res = activation->forward(th);
    std::cout << res.get<2>() << std::endl;

    delete activation;
    return 0;
}