#include <iostream>
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils/TensorHolder.h"
#include "layers/Activation.h"

using namespace std;
using namespace Eigen;


int main() {
    Tensor<double, 2> tensor(2, 3);
    tensor.setConstant(1.0);
    std::cout << tensor << std::endl;

    TensorHolder<double> my_tensor_holder(tensor);

    auto& retrieved_tensor = my_tensor_holder.get<2>();
    std::cout << retrieved_tensor << std::endl;

    auto relu = activations::ReLU<double>();

    return 0;
}