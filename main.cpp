#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include "models/Model.h"
#include "utils/Initializer.h"
#include "layers/Dense.h"
#include "utils/Optimizer.h"
#include "layers/Activation.h"

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
    std::string X_path = "../mnist/mnist_train_data.csv";
    std::string y_path = "../mnist/mnist_train_labels.csv";

    auto X_train = read_csv<60000, 784, 1>(X_path);
    auto y_train = read_csv<60000, 10, 1>(y_path);

    X_train /= X_train.constant(255);

    initializers::GlorotNormal<double> initializer;
    initializer.set_seed(42);

    Model<double, 3, 3> model("model", new optimizers::SGD<double>(0.05), new loss_functions::BinaryCrossEntropy<double>());

    auto input = model.addLayer<DenseLayer<double>>(784, 100, "dense 1", initializer);
    auto hidden = model.addLayer<DenseLayer<double>>(100, 50, "dense 2", initializer);
    auto hidden2 = model.addLayer<DenseLayer<double>>(50, 25, "dense 3", initializer);
    auto hidden3 = model.addLayer<DenseLayer<double>>(25, 10, "dense 3", initializer);
    auto sigmoid = model.addLayer<activations::Sigmoid<double, 2>>();
    auto sigmoid_2 = model.addLayer<activations::Sigmoid<double, 2>>();
    auto sigmoid_3 = model.addLayer<activations::Sigmoid<double, 2>>();
    auto out = model.addLayer<activations::Softmax<double, 2>>();

    connectLayers(input, sigmoid);
    connectLayers(sigmoid, hidden);
    connectLayers(hidden, sigmoid_2);
    connectLayers(sigmoid_2, hidden2);
    connectLayers(hidden2, sigmoid_3);
    connectLayers(sigmoid_3, hidden3);
    connectLayers(hidden3, out);

    model.setInput(input);
    model.setOut(out);


    std::cout << "Start:" << std::endl;
    model.fit(X_train, y_train, 20, 200, 4);
    model.test(X_train, y_train);

    return 0;
}