#include <iostream>
#include <utils/TensorHolder.h>
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "layers/Dense.h"
#include "utils/RandomNormal.h"
#include "utils/Optimizer.h"
#include <chrono>

using namespace std;
using namespace Eigen;

std::vector<double> func(){
    return std::vector<double>{3};
}

double func2(const std::vector<double>& vect){
    return vect[0];
}

int main() {
    func2(func());
    optimizers::SGD<double> sgd(2.0);
//    Optimizer<double>* sgd = new optimizers::SGD<double>(1.0);
    const auto* sgd_optimizer = new optimizers::SGD<double>(1.0);
    std::vector<const Optimizer<double>*> vct;
    vct.push_back(sgd_optimizer);
    RandomNormal<double> rn {};

    DenseLayer<double> dl(100, 100, "1", rn);
    Tensor<double, 2> X(100, 1);
    Tensor<double, 2> grads(100, 100);
    grads.setRandom();
    dl.forward(TensorHolder<double>(X));
//    dl.backward(TensorHolder(grads), sgd);

    return 0;
}