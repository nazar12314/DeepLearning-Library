#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "models/Model.h"
#include "utils/Initializer.h"
#include "layers/Dense.h"
#include "utils/Optimizer.h"
#include "layers/Activation.h"
#include "utils/MnistDataset.h"

int main() {

    MnistDataset<double> mnst;
    Tensor<double, 3> training_labels = mnst.get_training_labels();
    Tensor<double, 3> training_data = mnst.get_training_images();

    initializers::GlorotNormal<double> initializer;
    initializer.set_seed(42);


    Model<double, 3, 3> model("model", new optimizers::SGD<double>(0.001), new loss_functions::MSE<double>());

    DenseLayer<double> layer (748, 100, "dense 1", initializer);
    DenseLayer<double> layer2 (100, 50, "dense 2", initializer);
    DenseLayer<double> layer3 (50, 10, "dense 2", initializer);
    activations::ReLU<double, 2> relu;
    activations::ReLU<double, 2> relu2;
    auto input = model.addLayer(layer);
    auto hidden = model.addLayer(layer2);
    auto hidden2 = model.addLayer(layer3);
    auto relu_ = model.addLayer(relu);
    auto relu__ = model.addLayer(relu2);
    connectLayers(input, relu_);
    connectLayers(relu_, hidden);
    connectLayers(hidden, relu__);
    connectLayers(relu__, hidden2);

    model.setInput(input);
    model.setOut(hidden2);

    std::cout << "Start:" << std::endl;

    const size_t batch_size = 100;
    Eigen::array<size_t, 3> batch_shape{batch_size,
                                        size_t(training_data.dimension(1)),
                                        1};
    Eigen::array<size_t, 3> batch_shape_y{batch_size,
                                        size_t(training_labels.dimension(1)),
                                        1};
    model.test(training_data, training_labels);
    for (int j=0; j<5; ++j){
        for (size_t i=0; i<training_data.dimension(0); i+=batch_size){
            Tensor<double, 3> X = training_data.slice(Eigen::array<size_t , 3>({i, 0, 0}), batch_shape);
            Tensor<double, 3> y = training_labels.slice(Eigen::array<size_t , 3>({i, 0, 0}), batch_shape_y);
//        std::cout << y.dimension(0) << " " << y.dimension(1) << " " << y.dimension(2) << std::endl;
            model.fit(X, y, 1);
        }
        model.test(training_data, training_labels);
    }

//    Tensor<double, 3> data(4, 3, 1);
//    data.setValues({
//                           {{1}, {2}, {3}},
//                           {{4}, {5}, {6}},
//                           {{7}, {8}, {9}},
//                           {{10}, {11}, {12}}
//    });
//    Tensor<double, 2> w(3, 5);
//    w.setValues({
//                        {1, 2, 3, 4, 5},
//                        {6, 7, 8, 9, 10},
//                        {11, 12, 13, 14, 15}
//    });
////    std::cout << w.shuffle(Eigen::array<int, 2>{1, 0});
//    Tensor<double, 3> res = data.contract(w.shuffle(Eigen::array<int, 2>{1, 0}), Eigen::array<Eigen::IndexPair<int>, 1> {Eigen::IndexPair<int>(1, 1)});
//    res = res.reshape(Eigen::array<int, 3>({4, 5, 1}));
//    std::cout << res.dimension(0) << " " << res.dimension(1) << " " << res.dimension(2) << std::endl << std::endl;
//    std::cout << res;

//    Tensor<double, 3> sh = data.shuffle(Eigen::array<int, 3>{0, 2, 1});
//    std::cout << sh.dimension(0) << " " << sh.dimension(1) << " " << sh.dimension(2) << " " << std::endl;
//    Tensor<double, 3> data_broad = data.broadcast(Eigen::array<int, 3>{1, 1, 3}).reshape(Eigen::array<int, 3>{4, 3, 3});
//    Tensor<double, 3> data2(4, 1, 3);
//    data2.setValues({
//                           {{1}, {2}, {3}},
//                           {{4}, {5}, {6}},
//                           {{7}, {8}, {9}},
//                           {{10}, {11}, {12}}
//                   });
//    Tensor<double, 3> data_broad2 = data2.broadcast(Eigen::array<int, 3>{1, 3, 1}).reshape(Eigen::array<int, 3>{4, 3, 3});
//    int a = 2;
//    int b = 2;
//    int c = 2;
//    int d = 2;
//    Eigen::Tensor<double, 2> reshaped1 = data.reshape(Eigen::array<int, 2>{4, 3});
//    Eigen::Tensor<double, 2> reshaped2 = data2.reshape(Eigen::array<int, 2>{4, 3});

//    Tensor<double, 3> out  = reshaped1.contract(reshaped2, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)});
//    Tensor<double, 3> out =  data.contract(data2, Eigen::array<Eigen::IndexPair<int>, 2> {Eigen::IndexPair<int>(a, b), Eigen::IndexPair<int>(c, d)}) ;
//    Tensor<double, 4> out = data_broad2.contract(data, Eigen::array<Eigen::IndexPair<int>, 2>{Eigen::IndexPair<int>(2, 2)});
//    std::cout << out.dimension(0) << " " << out.dimension(1) << " " << out.dimension(2) << " " << out.dimension(3) << std::endl;
//    std::cout << out;

//    Tensor<double, 3> res = data_broad * data_broad2;
//    std::cout << res.dimension(0) << " " << res.dimension(1) << " " << res.dimension(2) << std::endl;
//    std::cout << res.mean(Eigen::array<int, 1>{0});

    return 0;
}
