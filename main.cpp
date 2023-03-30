#include <iostream>
#include <chrono>
//#include "eigen/Eigen/Eigen"
#include "eigen3/unsupported/Eigen/CXX11/Tensor"

using namespace std;
using namespace Eigen;

#define n 2000

int main() {
//    Eigen::MatrixXd A_ = Eigen::MatrixXd::Random(n, n);
    Tensor<double, 2> A(n, n);
    A.setRandom();
//    Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, n);
//    Eigen::MatrixXd D;
    Tensor<double, 2> B(n, n);
    B.setRandom();
    Tensor<double, 2> D(n, n);

    auto start_time = std::chrono::high_resolution_clock::now();

    D = A * B;

    cout << D(0, 0) << endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    cout << "Threads: " << Eigen::nbThreads() << endl;
    cout << "Time: " << duration_ms << endl;

    return 0;
}


//g++ -framework Accelerate -DEIGEN_USE_BLAS -ftree-vectorize main.cpp -Wno-deprecated-declarations -DNDEBUG -I./eigen -O3 -o main
