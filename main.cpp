#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "utils/Dataset.h"


int main() {
    Dataset<double, double, 3, 3> dataset({3, 4, 5}, {3, 4, 5});
    dataset.read_from_files("eigen_tensor.txt", "eigen_tensor.txt");
    std::cout << dataset.X_train.dimensions() << std::endl;
}
