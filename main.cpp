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

    Tensor<double, 2> ts1 (3, 1);
    ts1.setRandom();
    TensorHolder<double> th1 (ts1);

    Tensor<double, 2> ts2 (3, 1);
    ts2.setRandom();
    TensorHolder<double> th2 (ts2);
//    std::cout<<ts1<<std::endl<<std::endl<<ts2<<std::endl<<std::endl;
//    auto error = (ts1 - ts2).pow(2).mean();
//    std::cout<<error;
//    return 0;
//
//    Tensor<double, 2> differ = (ts1-ts2);
//    Tensor<double, 2> error = differ*differ.constant(2.0f/differ.dimension(0));
//    std::cout<<error;
//    return 0;
    Loss<double>* l = new loss_functions::MSE<double>();
    std::cout<<ts1<<std::endl<<std::endl<<ts2<<std::endl<<std::endl;
    std::cout<<l->get_error(th1, th2)<<std::endl;
    std::cout<<l->get_error_der(th1, th2).template get<2>();
    return 0;
}