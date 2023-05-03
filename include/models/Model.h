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

template<typename T, size_t Dim>
struct NodeTriplet {
    function_node<std::pair<Tensor<T, Dim+1, Eigen::RowMajor>, int>, std::pair<Tensor<T, Dim+1, Eigen::RowMajor>, int>> forward;
    function_node<Tensor<T, Dim+1, Eigen::RowMajor>, Tensor<T, Dim+1, Eigen::RowMajor>> test;
    function_node<std::pair<Tensor<T, Dim+1, Eigen::RowMajor>, int>, std::pair<Tensor<T, Dim+1, Eigen::RowMajor>, int>> backward;
};

template<class T, size_t InpDim, size_t OutDim>
class Model {
    std::string name;
    Optimizer<T> *optimizer;
    Loss<T> *loss;
    graph flowGraph;
    broadcast_node<std::pair<Tensor<T, InpDim, Eigen::RowMajor>, int>> modelInputNode;
    broadcast_node<Tensor<T, InpDim, Eigen::RowMajor>> modelTestNode;
    broadcast_node<std::pair<Tensor<T, OutDim, Eigen::RowMajor>, int>> modelOutNode;
    Tensor<T, OutDim, Eigen::RowMajor> modelOutput;
    std::pair<Tensor<T, OutDim, Eigen::RowMajor>, int> prediction;
    function_node<Tensor<T, OutDim, Eigen::RowMajor>> saveOutNode;
    function_node<std::pair<Tensor<T, OutDim, Eigen::RowMajor>, int>> pushOutNode;
    tbb::concurrent_queue<std::pair<Tensor<T, OutDim, Eigen::RowMajor>, int>> out_queue;
    
public:
    Model(const std::string &name_, const Optimizer<T> *optimizer_, const Loss<T> *loss_) :
        name(name_),
        optimizer((Optimizer<T> *) optimizer_),
        loss((Loss<T> *) loss_),
        modelInputNode(flowGraph),
        modelTestNode(flowGraph),
        modelOutNode(flowGraph),
        saveOutNode(flowGraph, 1, [&](const Tensor<T, OutDim, Eigen::RowMajor> &output) {
                        modelOutput = output;
        }),
        pushOutNode(flowGraph, 1, [&](std::pair<Tensor<T, OutDim, Eigen::RowMajor>, int> output) {
            out_queue.push(std::move(output));
        }) {};

    void setInput(NodeTriplet<T, InpDim - 1> &node) {
        make_edge(modelInputNode, node.forward);
        make_edge(modelTestNode, node.test);
    }

    void setOut(NodeTriplet<T, OutDim - 1> &node) {
        make_edge(node.forward, pushOutNode);
        make_edge(node.test, saveOutNode);
        make_edge(modelOutNode, node.backward);
    }

    template<size_t Dim>
    NodeTriplet<T, Dim> addLayer(Layer<T, Dim> &layer) {
        auto forward_func = [&layer](std::pair<Tensor<T, Dim+1, Eigen::RowMajor>, int> inputs) -> std::pair<Tensor<T, Dim+1, Eigen::RowMajor>, int> {
            return std::make_pair(std::move(layer.forward(inputs.first, inputs.second)), inputs.second);
        };
        auto test_func = [&layer](const Tensor<T, Dim+1, Eigen::RowMajor> &inputs) -> Tensor<T, Dim+1, Eigen::RowMajor> {
            return layer.forward(inputs, false);
        };
        auto backward_func = [&layer, this](std::pair<Tensor<T, Dim+1, Eigen::RowMajor>, int> grads) -> std::pair<Tensor<T,
                Dim + 1, Eigen::RowMajor>, int> {
            return std::make_pair(std::move(layer.backward(grads.first, *(this->optimizer), grads.second)),
                                  grads.second);
        };

        auto node_forward = function_node < std::pair<Tensor<T, Dim+1, Eigen::RowMajor>, int>, std::pair<Tensor<T, Dim+1, Eigen::RowMajor>, int>>
        (flowGraph, unlimited, forward_func);
        auto node_test = function_node < Tensor<T, Dim+1, Eigen::RowMajor>, Tensor<T, Dim+1, Eigen::RowMajor>>
        (flowGraph, unlimited, test_func);
        auto node_backward = function_node <std::pair<Tensor<T, Dim+1, Eigen::RowMajor>,
        int> , std::pair<Tensor<T, Dim+1, Eigen::RowMajor>, int >> (flowGraph, unlimited, backward_func);

        return NodeTriplet<T, Dim>{node_forward, node_test, node_backward};
    }

    auto predict(const Tensor<T, InpDim, Eigen::RowMajor> &inputs, const int n_minibathes, bool train = true) {
        size_t minibatch_size = inputs.dimension(0) / n_minibathes;
        Eigen::array<size_t, 3> mini_batch_shape{minibatch_size, size_t(inputs.dimension(1)), 1};

        for (size_t i = 0; i < n_minibathes; i++) {
            modelInputNode.try_put(std::make_pair(
                    std::move(inputs.slice(Eigen::array<size_t, 3>({i * minibatch_size, 0, 0}), mini_batch_shape)),
                    int(i)));
        }
//        modelInputNode.try_put(inputs);
        flowGraph.wait_for_all();
//        return modelOutput;
    }

    void backward(const Tensor<T, OutDim, Eigen::RowMajor> &labels, const int n_minibathes = 1) {
        size_t minibatch_size = labels.dimension(0) / n_minibathes;
        Eigen::array<size_t, 3> mini_batch_shape{minibatch_size, size_t(labels.dimension(1)), 1};
        for (size_t i = 0; i < n_minibathes; i++) {
            out_queue.try_pop(prediction);
            Tensor<T, OutDim, Eigen::RowMajor> grads = loss->calculate_grads(prediction.first, labels.slice(
                    Eigen::array<size_t, 3>({i * minibatch_size, 0, 0}), mini_batch_shape));
//            std::cout << "before put\n";
            modelOutNode.try_put(std::make_pair(std::move(grads), prediction.second));
        }
//        std::cout << "after loop\n";
        flowGraph.wait_for_all();
//        std::cout << "after loop2\n";
    }

    void test(const Tensor<T, InpDim, Eigen::RowMajor> &inputs, const Tensor<T, OutDim, Eigen::RowMajor> &labels) {
        modelTestNode.try_put(inputs);
        flowGraph.wait_for_all();
        std::cout << "Loss: " << loss->calculate_loss(modelOutput, labels) << std::endl;
        double num_equal_examples = 0;
        for (int i = 0; i < modelOutput.dimension(0); ++i) {
            Tensor<bool, 0, Eigen::RowMajor> equal = ((modelOutput.chip(i, 0).argmax() == labels.chip(i, 0).argmax()));
            if (equal(0)) {
                num_equal_examples++;
            }
        }
        std::cout << "Accuracy: " << num_equal_examples << " / " << labels.dimension(0) << " : "
                  << num_equal_examples / labels.dimension(0) * 100 << "%" << std::endl;
    }

    void
    fit(const Tensor<T, InpDim, Eigen::RowMajor> &inputs, const Tensor<T, OutDim, Eigen::RowMajor> &labels, const size_t epochs, const int batch_size,
        const int n_minibathes) {
        Eigen::array<int, 3> batch_shape{batch_size, int(inputs.dimension(1)), 1};
        Eigen::array<int, 3> batch_shape_y{batch_size, int(labels.dimension(1)), 1};
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            auto start_time = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < inputs.dimension(0); i += batch_size) {
                predict(inputs.slice(Eigen::array<int, 3>({i, 0, 0}), batch_shape), n_minibathes);
                backward(labels.slice(Eigen::array<int, 3>({i, 0, 0}), batch_shape_y), n_minibathes);
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
            std::cout << "Time: " << duration.count() << std::endl;
            flowGraph.wait_for_all();
        }
    }
};

template<class T, size_t Dim1, size_t Dim2>
void connectLayers(NodeTriplet<T, Dim1> &start, NodeTriplet<T, Dim2> &end) {
    make_edge(start.forward, end.forward);
    make_edge(start.test, end.test);
    make_edge(end.backward, start.backward);
}

#endif //NEURALIB_MODEL_H
