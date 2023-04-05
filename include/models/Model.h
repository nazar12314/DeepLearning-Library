//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_MODEL_H
#define NEURALIB_MODEL_H

#include <string>
#include <list>
#include <vector>
#include "layers/Layer.h"
#include "utils/Optimizer.h"
#include "utils/Loss.h"

template <class T>
class Model {
    std::string name;
    std::list<Layer<T>*> layers;
    Optimizer<T>* optimizer;
    Loss<T>* loss;

public:
    Model(const std::string& name_, const Optimizer<T>* optimizer_, const Loss<T>* loss_):
        name(name_),
        optimizer((Optimizer<T> *) optimizer_),
        loss((Loss<T> *) loss_)
    {};

    void addLayer(Layer<T>* layer) {
        layers.push_back(layer);
    };

    TensorHolder<T> predict(const TensorHolder<T>& input) {
        TensorHolder<T> output = input;

        for (auto& layer : layers) {
            output = layer -> forward(output);
        }

        return output;
    }

    void fit(const TensorHolder<T>& inputs, const TensorHolder<T>& labels, const size_t epochs) {
        const Tensor<T, 3>& inputs_unpacked = inputs.template get<3>();
        const Tensor<T, 3>& labels_unpacked = labels.template get<3>();
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double error = 0;
            for (int i=0; i < inputs_unpacked.dimension(0); ++i){
                TensorHolder<T> instance{Tensor<T, 2>{inputs_unpacked.chip(i, 0)}};
                TensorHolder<T> instance_label{Tensor<T, 2>{labels_unpacked.chip(i, 0)}};
                TensorHolder<T> output = predict(instance);
                error += loss->calculate_loss(output, instance_label)(0);
                TensorHolder<T> grads = loss->calculate_grads(output, instance_label);

                for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                    grads = (*it) -> backward(grads, *optimizer);
                }
            }
            std::cout << "Epoch: " << epoch << "; Loss: " << error/epochs << std::endl;
        }
    }
};


#endif //NEURALIB_MODEL_H
