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
    Optimizer<T> optimizer;
    Loss<T> loss;

public:
    Model(const std::string& name_, Optimizer<T>& optimizer_, Loss<T>& loss_):
        name(name_),
        optimizer(optimizer_),
        loss(loss_)
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
        for (int epoch = 0; epoch < epochs; ++epoch) {

            TensorHolder<T> output = predict(inputs);
            int error = loss.calculate_loss(output, labels);

            TensorHolder<T> grads = loss.calculate_grads(output, labels);

            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                grads = (*it) -> backward(grads, optimizer);
            }

            std::cout << "Epoch: " << epoch << "; Loss: " << error << std::endl;
        }
    }
};


#endif //NEURALIB_MODEL_H
