//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_MODEL_H
#define NEURALIB_MODEL_H

#include <string>
#include <list>
#include <vector>
#include <variant>
#include "layers/Layer.h"
#include "utils/Optimizer.h"
#include "utils/Loss.h"

template <class T>
using layer_variant_ptr = std::variant<Layer<T, 2>*, Layer<T, 3>*>;

template <class T>
using tensor_variant_ref = std::variant<Tensor<T, 2>&, Tensor<T, 3>&>;

template <class T>
using tensor_variant = std::variant<Tensor<T, 2>, Tensor<T, 3>>;

template <class T>
class Model {
    std::string name;
    std::list<layer_variant_ptr<T>> layers;
    Optimizer<T>* optimizer;
    Loss<T>* loss;

public:
    Model(const std::string& name_, const Optimizer<T>* optimizer_, const Loss<T>* loss_):
        name(name_),
        optimizer((Optimizer<T>*) optimizer_),
        loss((Loss<T> *) loss_)
    {};

    void addLayer(layer_variant_ptr<T> layer) {
        layers.push_back(layer);
    };

    tensor_variant<T> predict(const tensor_variant<T>& input) {
        tensor_variant<T> output = input;

        for (auto& layer : layers) {
            std::visit([&layer, &output](auto&& arg) {
//                using W = std::decay<decltype(arg)>;

//                if constexpr (std::is_same_v<W, Tensor<T, 2>> || std::is_same_v<W, Tensor<T, 3>>) {
                std::visit([&output, &arg](auto&& arg_layer){
//                    using L = std::decay<decltype(arg_layer)>;
//                    if constexpr (std::is_same_v<L, Layer<T, 2>> || std::is_same_v<L, Layer<T, 3>>){
                    output = arg_layer -> forward(arg);
//                    }
//                    else {
//                        throw std::runtime_error("Layer ban!");
//                    }
                }, layer);
//                }
//                else {
//                    throw std::runtime_error("No such dimension!");
//                }
            }, output);
        }

        return output;
    }

//    void fit(const TensorHolder<T>& inputs, const TensorHolder<T>& labels, const size_t epochs) {
//        const Tensor<T, 3>& inputs_unpacked = inputs.template get<3>();
//        const Tensor<T, 3>& labels_unpacked = labels.template get<3>();
//        for (int epoch = 0; epoch < epochs; ++epoch) {
//            double error = 0;
//            size_t input_size = inputs_unpacked.dimension(0);
//            for (size_t i = 0; i < input_size; ++i){
//                TensorHolder<T> instance{Tensor<T, 2>{inputs_unpacked.chip(i, 0)}};
//                TensorHolder<T> instance_label{Tensor<T, 2>{labels_unpacked.chip(i, 0)}};
//                TensorHolder<T> output = predict(instance);
//                double loss_ = loss->calculate_loss(output, instance_label)(0);
//                error += loss_;
//                TensorHolder<T> grads = loss->calculate_grads(output, instance_label);
//                std::cout << i << " / " << input_size << " | loss: " << loss_ << " | mingrad: " << grads.template get<2>().minimum() << " | maxgrad: " << grads.template get<2>().maximum() << std::endl;
//                for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
//                    grads = (*it) -> backward(grads, *optimizer);
//                }
//            }
//            std::cout << "Epoch: " << epoch << "; Loss: " << error / epochs << std::endl;
//        }
//    }
};


#endif //NEURALIB_MODEL_H
