#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include "models/Model.h"
#include "utils/Initializer.h"
#include "layers/Dense.h"
#include "utils/Optimizer.h"
#include "layers/Activation.h"
#include <layers/Convolution.h>

#include <fstream>
#include <sstream>
#include <map>

template <size_t ROWS, size_t COLS, size_t CHNALLES>
Tensor<double, 3, Eigen::RowMajor> read_csv(const std::string& filename){
    Tensor<double, 3, Eigen::RowMajor> data(ROWS, COLS, CHNALLES);

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::logic_error("Cannot open file of labels");
    }
    std::string line;
    int row = 0;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        int col = 0;
        while (getline(ss, cell, ',')) {
            float value = stof(cell);
            data(row, col, 0) = value;
            col++;
        }
        row++;
    }

    return data;
}

int main() {
//    std::string X_path = "../mnist/mnist_train_data.csv";
//    std::string y_path = "../mnist/mnist_train_labels.csv";
//
//    auto X_train = read_csv<60000, 784, 1>(X_path);
//    auto y_train = read_csv<60000, 10, 1>(y_path);
//
//    X_train /= X_train.constant(255);
//
//    initializers::GlorotNormal<double> initializer;
//    initializer.set_seed(42);
//
//    Model<double, 3, 3> model("model", new optimizers::SGD<double>(0.05), new loss_functions::BinaryCrossEntropy<double>());
//
//    auto input = model.addLayer<DenseLayer<double>>(784, 20, "dense 1", initializer);
//    auto hidden = model.addLayer<DenseLayer<double>>(20, 20, "dense 2", initializer);
//    auto hidden2 = model.addLayer<DenseLayer<double>>(20, 20, "dense 3", initializer);
//    auto hidden3 = model.addLayer<DenseLayer<double>>(20, 10, "dense 3", initializer);
//    auto skip = model.addSkip(hidden, hidden2);
//    auto sigmoid = model.addLayer<activations::Sigmoid<double, 2>>();
//    auto sigmoid_2 = model.addLayer<activations::Sigmoid<double, 2>>();
//    auto sigmoid_3 = model.addLayer<activations::Sigmoid<double, 2>>();
//    auto out = model.addLayer<activations::Softmax<double, 2>>();
//
//    connect(input, sigmoid);
//    connect(sigmoid, hidden);
//    connect(hidden, sigmoid_2);
//    connect(sigmoid_2, hidden2);
//    // hidden2 automatically connected to skip
//    connect(skip, sigmoid_3);
//    connect(sigmoid_3, hidden3);
//    connect(hidden3, out);
//
//    model.setInput(input);
//    model.setOut(out);
//
//    std::cout << "Start:" << std::endl;
//    model.fit(X_train, y_train, 10, 200, 4);

    size_t kernel_size = 2, kernel_depth = 2, input_depth = 2, batch_amount = 1;
////
    Tensor<double, 4, Eigen::RowMajor> inputTensor (batch_amount, 6, 6, input_depth);

//    inputTensor.setValues({
//                                  {{{1, 1}, {2, 2}, {3, 3}}, {{4, 4}, {5, 5}, {6, 6}}, {{7, 7}, {8, 8}, {9, 9}}},
//    });

    inputTensor.setRandom();

    initializers::GlorotNormal<double> initializer;
    ConvolutionLayer<double> layer (6, 6, input_depth, kernel_size, kernel_depth, "Conv1", initializer);
    MaxPooling<double, 3> max_pool(2);
    FlattenLayer<double> flatten;

    Tensor<double, 4, Eigen::RowMajor> res = layer.forward(inputTensor);
    std::cout << res.dimensions() << std::endl;

    Tensor<double, 4, Eigen::RowMajor> maxpool_res = max_pool.forward(res);
    std::cout << maxpool_res.dimensions() << std::endl;
    auto flatten_res = flatten.forward(maxpool_res);
    std::cout << flatten_res.dimensions() << std::endl;
    auto back = flatten.backward(flatten_res);
    std::cout << back.dimensions();

    std::cout << maxpool_res << std::endl << std::endl;
    std::cout << back << std::endl;

    return 0;
}