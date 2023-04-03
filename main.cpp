#include <iostream>
#include <utils/TensorHolder.h>
#include "eigen3/unsupported/Eigen/CXX11/Tensor"
#include "layers/Activation.h"

using namespace std;
using namespace Eigen;


int main() {
    Tensor<double, 2> tensor(2, 3);
    tensor.setConstant(5.0);

    TensorHolder<double> my_tensor_holder(tensor);

    auto res = activations::relu_function(my_tensor_holder);
    auto res_prime = activations::relu_function_prime(my_tensor_holder);

    auto res_holder = res.template get<2>();
    auto res_holder_prime = res_prime.template get<2>();

    std::cout << res_holder << std::endl;
    std::cout << res_holder_prime << std::endl;

    return 0;
}