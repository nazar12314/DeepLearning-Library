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
#include <chrono>
#include <tbb/concurrent_queue.h>

using namespace tbb::flow;

// pair of function_nodes, which are forward and backward
//template<class T, size_t Dim>
//using layer_node = std::pair<function_node<Tensor<T, Dim>, Tensor<T, Dim>>, function_node<Tensor<T, Dim>, Tensor<T, Dim>>>;

template <typename T, size_t Dim>
struct NodeTriplet {
    function_node<std::pair<Tensor<T, Dim+1>, int>, std::pair<Tensor<T, Dim+1>, int>> forward;
    function_node<Tensor<T, Dim+1>, Tensor<T, Dim+1>> test;
    function_node<std::pair<Tensor<T, Dim+1>, int>, std::pair<Tensor<T, Dim+1>, int>> backward;
};

template <class T, size_t InpDim, size_t OutDim>
class Model {
    std::string name;
    Optimizer<T>* optimizer;
    Loss<T>* loss;
    graph flowGraph;
    broadcast_node<std::pair<Tensor<T, InpDim>, int>> modelInputNode;
    broadcast_node<Tensor<T, InpDim>> modelTestNode;
    broadcast_node<std::pair<Tensor<T, OutDim>, int>> modelOutNode;
    Tensor<T, OutDim> modelOutput;
    std::pair<Tensor<T, OutDim>, int> prediction;
    function_node<Tensor<T, OutDim>> saveOutNode;
    function_node<std::pair<Tensor<T, OutDim>, int>> pushOutNode;
    tbb::concurrent_queue<std::pair<Tensor<T, OutDim>, int>> out_queue;


public:
    Model(const std::string& name_, const Optimizer<T>* optimizer_, const Loss<T>* loss_):
            name(name_),
            optimizer((Optimizer<T>*) optimizer_),
            loss((Loss<T> *) loss_),
            modelInputNode(flowGraph),
            modelTestNode(flowGraph),
            modelOutNode(flowGraph),
            saveOutNode(flowGraph, 1, [&](const Tensor<T, OutDim>& output){
                modelOutput = output;
            }),
            pushOutNode(flowGraph, 1, [&](std::pair<Tensor<T, OutDim>, int> output){
                out_queue.push(std::move(output));
            })
    {};

    void setInput(NodeTriplet<T, InpDim-1>& node){
        make_edge(modelInputNode, node.forward);
        make_edge(modelTestNode, node.test);
    }

    void setOut(NodeTriplet<T, OutDim-1>& node){
        make_edge(node.forward, pushOutNode);
        make_edge(node.test, saveOutNode);
        make_edge(modelOutNode, node.backward);
    }

    template<size_t Dim>
    NodeTriplet<T, Dim> addLayer(Layer<T, Dim>& layer){
        auto forward_func = [&layer](std::pair<Tensor<T, Dim+1>, int> inputs) -> std::pair<Tensor<T, Dim+1>, int> {
            return std::make_pair(std::move(layer.forward(inputs.first, inputs.second)), inputs.second);
        };
        auto test_func = [&layer](const Tensor<T, Dim+1>& inputs) -> Tensor<T, Dim+1> {
            return layer.forward(inputs, false);
        };
        auto backward_func = [&layer, this](std::pair<Tensor<T, Dim+1>, int> grads) -> std::pair<Tensor<T, Dim+1>, int> {
            return std::make_pair(std::move(layer.backward(grads.first, *(this->optimizer), grads.second)), grads.second);
        };

        auto node_forward = function_node<std::pair<Tensor<T, Dim+1>, int>, std::pair<Tensor<T, Dim+1>, int>>(flowGraph, unlimited, forward_func);
        auto node_test = function_node<Tensor<T, Dim+1>, Tensor<T, Dim+1>>(flowGraph, unlimited, test_func);
        auto node_backward = function_node<std::pair<Tensor<T, Dim+1>, int>, std::pair<Tensor<T, Dim+1>, int>>(flowGraph, unlimited, backward_func);

        return NodeTriplet<T, Dim>{node_forward, node_test, node_backward};
    }

    auto predict(const Tensor<T, InpDim>& inputs, const int n_minibathes, bool train = true){
        size_t minibatch_size = inputs.dimension(0) / n_minibathes;
        Eigen::array<size_t , 3> mini_batch_shape{minibatch_size, size_t(inputs.dimension(1)), 1};

        for (size_t i=0; i<n_minibathes; i++){
            modelInputNode.try_put(std::make_pair(std::move(inputs.slice(Eigen::array<size_t , 3>({i*minibatch_size, 0, 0}), mini_batch_shape)), int(i)));
        }
//        modelInputNode.try_put(inputs);
        flowGraph.wait_for_all();
//        return modelOutput;
    }

    void backward(const Tensor<T, OutDim>& labels, const int n_minibathes = 1){
//        std::cout << "start of backward\n";
        size_t minibatch_size = labels.dimension(0) / n_minibathes;
        Eigen::array<size_t , 3> mini_batch_shape{minibatch_size, size_t(labels.dimension(1)), 1};
        for (size_t i=0; i<n_minibathes; i++){
//            std::cout << "before move\n";
//            Tensor<T, OutDim> prediction = std::move(out_queue.front());
//            std::cout << prediction.dimension(0) << std::endl;
//            std::cout << "before pop\n";
            out_queue.try_pop(prediction);
//            std::cout << "before grads\n";
            Tensor<T, OutDim> grads = loss->calculate_grads(prediction.first, labels.slice(Eigen::array<size_t , 3>({i*minibatch_size, 0, 0}), mini_batch_shape));
//            std::cout << "before put\n";
            modelOutNode.try_put(std::make_pair(std::move(grads), prediction.second));
        }
//        std::cout << "after loop\n";
        flowGraph.wait_for_all();
//        std::cout << "after loop2\n";
    }

    void test(const Tensor<T, InpDim>& inputs, const Tensor<T, OutDim>& labels){
        modelTestNode.try_put(inputs);
        flowGraph.wait_for_all();
        std::cout << "Loss: " << loss->calculate_loss(modelOutput, labels) << std::endl;
        double num_equal_examples = 0;
        for (int i = 0; i < modelOutput.dimension(0); ++i) {
            Tensor<bool, 0> equal = ((modelOutput.chip(i, 0).argmax() == labels.chip(i, 0).argmax()));
            if (equal(0)){
                num_equal_examples++;
            }
        }
        std::cout << "Accuracy: " << num_equal_examples <<" / "<< labels.dimension(0) << " : " << num_equal_examples/labels.dimension(0) * 100 << "%" << std::endl;
    }

    void fit(const Tensor<T, InpDim>& inputs, const Tensor<T, OutDim>& labels, const size_t epochs, const int batch_size,
             const int n_minibathes){
        Eigen::array<int, 3> batch_shape{batch_size, int(inputs.dimension(1)), 1};
        Eigen::array<int, 3> batch_shape_y{batch_size, int(labels.dimension(1)), 1};
        for (size_t epoch = 0; epoch < epochs; ++epoch){
            auto start_time = std::chrono::high_resolution_clock::now();
            for (int i=0; i<inputs.dimension(0); i+=batch_size){
//                Tensor<double, 3> X = inputs.slice(Eigen::array<int , 3>({i, 0, 0}), batch_shape);
//                Tensor<double, 3> y = labels.slice(Eigen::array<int , 3>({i, 0, 0}), batch_shape_y);
//                std::cout << "before predict\n";
                predict(inputs.slice(Eigen::array<int , 3>({i, 0, 0}), batch_shape), n_minibathes);
//                std::cout << "After predict:\n";
                backward(labels.slice(Eigen::array<int , 3>({i, 0, 0}), batch_shape_y), n_minibathes);
//                std::cout << "After backward\n";
//                Tensor<T, OutDim> grads = loss->calculate_grads(predicted, labels.slice(Eigen::array<int , 3>({i, 0, 0}), batch_shape_y));
//                modelOutNode.try_put(grads);
//                flowGraph.wait_for_all();
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
            std::cout << "Time: " << duration.count() << std::endl;
//            test(inputs, labels);
//            Tensor<T, OutDim> predicted = predict(inputs);
//            std::cout << "Loss: " << loss->calculate_loss(predicted, labels) << std::endl;
//            Tensor<T, OutDim> grads = loss->calculate_grads(predicted, labels);
//            std::cout << grads << std::endl;
//            backward(grads);
//            modelOutNode.try_put(grads);
//            flowGraph.wait_for_all();
        }
    }
};

template<class T, size_t Dim1, size_t Dim2>
void connectLayers(NodeTriplet<T, Dim1>& start, NodeTriplet<T, Dim2>& end){
    make_edge(start.forward, end.forward);
    make_edge(start.test, end.test);
    make_edge(end.backward, start.backward);
}

#endif //NEURALIB_MODEL_H
