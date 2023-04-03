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
//    std::string name;
    std::list<Layer<T>*> layers;
//    Optimizer<T> optimizer;
//    Loss<T> loss;

public:
//    Model(const std::string& name_, Optimizer<T>& optimizer_, Loss<T>& loss_):
//        name(name_),
//        optimizer(optimizer_),
//        loss(loss_)
//    {};

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
};


#endif //NEURALIB_MODEL_H
