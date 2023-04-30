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
#include <tbb/flow_graph.h>

using namespace tbb::flow;

// pair of function_nodes, which are forward and backward
template<class T, size_t Dim>
using layer_node = std::pair<function_node<Tensor<T, Dim>, Tensor<T, Dim>>, function_node<Tensor<T, Dim>, Tensor<T, Dim>>>;

template <class T, size_t InpDim, size_t OutDim>
class Model {
    std::string name;
    Optimizer<T>* optimizer;
    Loss<T>* loss;
    graph flowGraph;
    broadcast_node<Tensor<T, InpDim>> modelInputNode;
    broadcast_node<Tensor<T, OutDim>> modelOutNode;
    Tensor<T, OutDim> modelOutput;
    function_node<Tensor<T, OutDim>> saveOutNode;

public:
    Model(const std::string& name_, const Optimizer<T>* optimizer_, const Loss<T>* loss_):
            name(name_),
            optimizer((Optimizer<T>*) optimizer_),
            loss((Loss<T> *) loss_),
            modelInputNode(flowGraph),
            modelOutNode(flowGraph),
            saveOutNode(flowGraph, 1, [&](const Tensor<T, OutDim>& output){
                modelOutput = output;
            })
    {};

    void setInput(layer_node<T, InpDim>& node){
        make_edge(modelInputNode, node.first);
    }

    void setOut(layer_node<T, OutDim>& node){
        make_edge(node.first, saveOutNode);
        make_edge(modelOutNode, node.second);
    }

    template<size_t Dim>
    auto addLayer(Layer<T, Dim>& layer){
        auto forward_func = [&layer](const Tensor<T, Dim+1>& inputs) -> Tensor<T, Dim+1> {
            return layer.forward(inputs);
        };
        auto backward_func = [&layer, this](const Tensor<T, Dim+1>& grads) -> Tensor<T, Dim+1> {
            return layer.backward(grads, *(this->optimizer));
        };

        auto node_forward = function_node<Tensor<T, Dim+1>, Tensor<T, Dim+1>>(flowGraph, unlimited, forward_func);
        auto node_backward = function_node<Tensor<T, Dim+1>, Tensor<T, Dim+1>>(flowGraph, unlimited, backward_func);
        return std::make_pair(std::move(node_forward), std::move(node_backward));
    }

    auto predict(const Tensor<T, InpDim>& inputs){
        modelInputNode.try_put(inputs);
        flowGraph.wait_for_all();
        return modelOutput;
    }

    void backward(const Tensor<T, OutDim>& grads){
        const size_t batch_size = 600;
        Eigen::array<size_t, 3> batch_shape{batch_size,
                                            size_t(grads.dimension(1)),
                                            1};
        for (size_t i=0; i<grads.dimension(0); i+=batch_size){
            Tensor<T, OutDim> cut = grads.slice(Eigen::array<size_t , 3>({i, 0, 0}), batch_shape);
            modelOutNode.try_put(cut);
        }
        flowGraph.wait_for_all();
    }

    void test(const Tensor<T, InpDim>& inputs, const Tensor<T, OutDim>& labels){
        Tensor<T, OutDim> predicted = predict(inputs);
        std::cout << "\nLoss: " << loss->calculate_loss(predicted, labels) << std::endl;
        double num_equal_examples = 0;
        for (int i = 0; i < predicted.dimension(0); ++i) {
            Tensor<bool, 0> equal = ((predicted.chip(i, 0).argmax() == labels.chip(i, 0).argmax()));
            if (equal(0)){
                num_equal_examples++;
            }
        }
        std::cout << "Accuracy: " << num_equal_examples <<" / "<< labels.dimension(0) << " : " << num_equal_examples/labels.dimension(0) * 100 << "%" << std::endl;
    }

    void fit(const Tensor<T, InpDim>& inputs, const Tensor<T, OutDim>& labels, const size_t epochs){
        for (size_t epoch = 0; epoch < epochs; ++epoch){
            Tensor<T, OutDim> predicted = predict(inputs);
//            std::cout << "Loss: " << loss->calculate_loss(predicted, labels) << std::endl;
            Tensor<T, OutDim> grads = loss->calculate_grads(predicted, labels);
//            std::cout << grads << std::endl;
//            backward(grads);
            modelOutNode.try_put(grads);
            flowGraph.wait_for_all();
        }
    }
};

template<class T, int Dim>
void connectLayers(layer_node<T, Dim>& start, layer_node<T, Dim>& end){
    make_edge(start.first, end.first);
    make_edge(end.second, start.second);
}

#endif //NEURALIB_MODEL_H
