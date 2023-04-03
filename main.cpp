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
    optimizers::SGD<double> sgd(1.0);
    RandomNormal<double> rn {};
    auto start= std::chrono::high_resolution_clock::now();
    DenseLayer<double> dl(100, 100, "1", rn);
    TensorHolder<double> ts = dl.get_weights();
    std::cout << "First weight before optimization:\n" << ts.get<2>()(0, 0) << endl;
    sgd.apply_optimization(ts, dl);
    TensorHolder<double> ts2 = dl.get_weights();
    std::cout << "First weight after applying optimization with gradients same as layer weights(should result in zero):\n" << ts2.get<2>()(0, 0) << endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time: " << duration_ms;

    return 0;
}