#include <iostream>
#include <utils/TensorHolder.h>
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "layers/Dense.h"
#include "utils/Optimizer.h"
#include "utils/Loss.h"
#include <chrono>
#include "layers/Dense.h"
#include "utils/Initializer.h"
#include "models/Model.h"

using namespace Eigen;

int main() {
//    initializers::RandomNormal<double> ci (10);
//    Model<double> ml;
//
//    Tensor<double, 2> ts (10, 1);
//    ts.setRandom();
//
//    TensorHolder<double> th (ts);
//
//    ml.addLayer(new DenseLayer<double> (10, 5, "Dense 1", ci));
//    ml.addLayer(new DenseLayer<double> (5, 15, "Dense 2", ci));
//
//    std::cout << ml.predict(th).template get<2>();

    Tensor<double, 2> ts1 (10, 1);
    ts1.setRandom();
    TensorHolder<double> th1 (ts1);

    Tensor<double, 2> ts2 (10, 1);
    ts2.setRandom();
    TensorHolder<double> th2 (ts2);

    loss_functions::MSE<double>() l;
    return 0;
}