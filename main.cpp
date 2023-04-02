#include <iostream>
#include <utils/TensorHolder.h>
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "layers/Dense.h"
#include "utils/RandomNormal.h"


using namespace std;
using namespace Eigen;


int main() {
    RandomNormal<double> rn {};

    DenseLayer<double> dl(5, 2, "1", rn);

    Tensor<double, 2> X (5, 1);

    X.setRandom();

    Tensor<double, 2> forwarded = dl.forward(TensorHolder(X)).get<2>();

    std::cout << forwarded;

    return 0;
}