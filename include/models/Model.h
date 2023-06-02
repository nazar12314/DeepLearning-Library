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
#include "utils/tqdm.h"
#include <sstream>


using namespace tbb::flow;

template<class T, size_t Dim>
class SkipLayer;

template<class T>
class FlattenLayer;

/**
 * @brief Template struct representing a NodeTriplet.
 *
 * This struct represents a NodeTriplet, which consists of three function nodes and a function to retrieve a minibatch.
 * The NodeTriplet is templated on the type T and the size Dim.
 *
 * @tparam T The data type of the Tensor.
 * @tparam Dim The dimension of the Tensor.
 */
template<typename T, size_t Dim>
struct NodeTriplet {
    function_node<std::pair<Tensor<T, Dim + 1, Eigen::RowMajor>, int>, std::pair<Tensor<T, Dim + 1, Eigen::RowMajor>, int>> forward;
    function_node<Tensor<T, Dim + 1, Eigen::RowMajor>, Tensor<T, Dim + 1, Eigen::RowMajor>> test;
    function_node<std::pair<Tensor<T, Dim + 1, Eigen::RowMajor>, int>, std::pair<Tensor<T, Dim + 1, Eigen::RowMajor>, int>> backward;
    std::function<Tensor<double, Dim+1, Eigen::RowMajor>&(int)> get_minibatch;
};

/**
 * @brief Template struct representing a SkipNode.
 *
 * This struct represents a SkipNode, which consists of a forward function node, a backward broadcast node, and a test broadcast node.
 * The SkipNode is templated on the type T and the size Dim.
 *
 * @tparam T The data type of the Tensor.
 * @tparam Dim The dimension of the Tensor.
 */
template<typename T, size_t Dim>
struct SkipNode{
    function_node<std::pair<Tensor<T, Dim + 1, Eigen::RowMajor>, int>, std::pair<Tensor<T, Dim + 1, Eigen::RowMajor>, int>> forward;
    broadcast_node<std::pair<Tensor<T, Dim+1, Eigen::RowMajor>, int>> backward;
    broadcast_node<Tensor<T, Dim+1, Eigen::RowMajor>> test;
};

/**
 * @brief Template struct representing a FlattenNode.
 *
 * This struct represents a FlattenNode, which consists of forward, backward, and test function nodes.
 * The FlattenNode is templated on the type T.
 *
 * @tparam T The data type of the Tensor.
 */
template<typename T>
struct FlattenNode{
    function_node<std::pair<Tensor<T,  4, Eigen::RowMajor>, int>, std::pair<Tensor<T, 3, Eigen::RowMajor>, int>> forward;
    function_node<std::pair<Tensor<T,  3, Eigen::RowMajor>, int>, std::pair<Tensor<T, 4, Eigen::RowMajor>, int>> backward;
    function_node<Tensor<T, 4, Eigen::RowMajor>, Tensor<T, 3, Eigen::RowMajor>> test;
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
    /**
     * Constructor
     *
     * Constructs a Model object with the specified name, optimizer, and loss function.
     *
     * @param name_ The name of the model
     * @param optimizer_ Pointer to the optimizer object
     * @param loss_ Pointer to the loss function object
     */
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

    /**
     * setInput
     *
     * Sets the input node of the model.
     *
     * @param node The input node of type NodeTriplet<T, InpDim - 1>
     */
    void setInput(NodeTriplet<T, InpDim - 1> &node) {
        make_edge(modelInputNode, node.forward);
        make_edge(modelTestNode, node.test);
    }

    /**
     * setOut function
     *
     * Connects the output of a NodeTriplet to the model's output nodes.
     *
     * @tparam T The data type of the Tensor
     * @tparam OutDim The output dimension of the model
     * @param node The NodeTriplet whose output is connected to the model's output nodes
     */
    void setOut(NodeTriplet<T, OutDim - 1> &node) {
        make_edge(node.forward, pushOutNode);
        make_edge(node.test, saveOutNode);
        make_edge(modelOutNode, node.backward);
    }

    /**
     * addFlattenLayer function
     *
     * Adds a flatten layer to the model.
     *
     * @tparam T The data type of the Tensor
     * @return The FlattenNode representing the added flatten layer
     */
    FlattenNode<T> addFlattenLayer(){
        auto flatten = new FlattenLayer<T>();

        auto forward_func = [flatten](std::pair<Tensor<T, 4, Eigen::RowMajor>, int> inputs) -> std::pair<Tensor<T, 3, Eigen::RowMajor>, int> {
            return std::make_pair(std::move(flatten->forward(inputs.first, inputs.second)), inputs.second);
        };
        auto test_func = [flatten](const Tensor<T, 4, Eigen::RowMajor> &inputs) -> Tensor<T, 3, Eigen::RowMajor> {
            return flatten->forward(inputs, false);
        };
        auto backward_func = [flatten](std::pair<Tensor<T, 3, Eigen::RowMajor>, int> grads) -> std::pair<Tensor<T, 4, Eigen::RowMajor>, int> {
            return std::make_pair(std::move(flatten->backward(grads.first, grads.second)), grads.second);
        };
        auto node_forward = function_node<std::pair<Tensor<T, 4, Eigen::RowMajor>, int>, std::pair<Tensor<T, 3, Eigen::RowMajor>, int>> (flowGraph, unlimited, forward_func);
        auto node_test = function_node<Tensor<T, 4, Eigen::RowMajor>, Tensor<T, 3, Eigen::RowMajor>> (flowGraph, unlimited, test_func);
        auto node_backward = function_node<std::pair<Tensor<T, 3, Eigen::RowMajor>, int> , std::pair<Tensor< T, 4, Eigen::RowMajor>, int >> (flowGraph, unlimited, backward_func);

        return FlattenNode<T>{node_forward, node_backward, node_test};
    }

    /**
     * addLayer function
     *
     * Adds a layer to the model.
     *
     * @tparam LayerType The type of layer to add
     * @tparam Dim The dimension of the layer
     * @tparam Args The types of arguments to pass to the layer's constructor
     * @param args The arguments to pass to the layer's constructor
     * @return The NodeTriplet representing the added layer
     */
    template<typename LayerType, size_t Dim = 2, typename... Args>
    NodeTriplet<T, Dim> addLayer(Args&&... args) {

        auto layer = new LayerType(args...);

        auto forward_func = [layer](std::pair<Tensor<T, Dim + 1, Eigen::RowMajor>, int> inputs) -> std::pair<Tensor<T, Dim+ 1, Eigen::RowMajor>, int> {
            return std::make_pair(std::move(layer->forward(inputs.first, inputs.second)), inputs.second);
        };
        auto test_func = [layer](const Tensor<T, Dim + 1, Eigen::RowMajor> &inputs) -> Tensor<T, Dim + 1, Eigen::RowMajor> {
            return layer->forward(inputs, false);
        };
        auto backward_func = [layer, this](std::pair<Tensor<T, Dim + 1, Eigen::RowMajor>, int> grads) -> std::pair<Tensor<T, Dim + 1, Eigen::RowMajor>, int> {
            return std::make_pair(std::move(layer->backward(grads.first, *(this->optimizer), grads.second)), grads.second);
        };

        auto get_minibatch = [layer](int minibatchInd) -> Tensor<T, Dim + 1, Eigen::RowMajor>&{
            return layer->get_saved_minibatch(minibatchInd);
        };

        auto node_forward = function_node<std::pair<Tensor<T, Dim + 1, Eigen::RowMajor>, int>, std::pair<Tensor<T, Dim + 1, Eigen::RowMajor>, int>> (flowGraph, unlimited, forward_func);
        auto node_test = function_node<Tensor<T, Dim + 1, Eigen::RowMajor>, Tensor<T, Dim + 1, Eigen::RowMajor>> (flowGraph, unlimited, test_func);
        auto node_backward = function_node<std::pair<Tensor < T, Dim + 1, Eigen::RowMajor>, int > , std::pair<Tensor< T, Dim + 1, Eigen::RowMajor>, int >> (flowGraph, unlimited, backward_func);

        return NodeTriplet<T, Dim>{node_forward, node_test, node_backward, get_minibatch};
    }

    /**
     * addSkip function
     *
     * Adds a skip connection between two NodeTriplets in the model.
     *
     * @tparam T The data type of the Tensor
     * @tparam Dim The dimension of the NodeTriplets
     * @param node1 The first NodeTriplet
     * @param node2 The second NodeTriplet
     * @return The SkipNode representing the added skip connection
     */
    template<size_t Dim>
    SkipNode<T, Dim> addSkip(NodeTriplet<T, Dim> &node1, NodeTriplet<T, Dim> &node2){
        auto skip = new SkipLayer<T, Dim>(node1, node2);
        auto forward_func = [skip](std::pair<Tensor<T, Dim + 1, Eigen::RowMajor>, int> inputs) -> std::pair<Tensor<T, Dim+ 1, Eigen::RowMajor>, int> {
            return std::make_pair(std::move(skip->forward(inputs.first, inputs.second)), inputs.second);
        };

        auto node_forward = function_node<std::pair<Tensor<T, Dim + 1, Eigen::RowMajor>, int>, std::pair<Tensor<T, Dim + 1, Eigen::RowMajor>, int>> (flowGraph, unlimited, forward_func);
        broadcast_node<std::pair<Tensor<T, Dim+1, Eigen::RowMajor>, int>> backwardNode(flowGraph);
        broadcast_node<Tensor<T, Dim+1, Eigen::RowMajor>> testNode(flowGraph);

        auto node = SkipNode<T, Dim>{node_forward, backwardNode, testNode};

        make_edge(node.backward, node1.backward);
        make_edge(node.backward, node2.backward);
        make_edge(node2.forward, node.forward);
        make_edge(node2.test, node.test);

        return node;
    }

    /**
     * predict function
     *
     * Makes predictions using the model for the given inputs.
     *
     * @tparam T The data type of the Tensor
     * @tparam InpDim The input dimension of the model
     * @param inputs The input Tensor
     * @param n_minibathes The number of minibatches
     * @param train Flag indicating whether the model is in training mode
     */
    auto predict(const Tensor<T, InpDim, Eigen::RowMajor> &inputs, const int n_minibathes, bool train = true) {
        size_t minibatch_size = inputs.dimension(0) / n_minibathes;
        Eigen::array<size_t, 4> mini_batch_shape{minibatch_size, size_t(inputs.dimension(1)), size_t(inputs.dimension(2)), size_t(inputs.dimension(3))};

        for (size_t i = 0; i < n_minibathes; i++) {
            modelInputNode.try_put(std::make_pair(
                    std::move(inputs.slice(Eigen::array<size_t, 4>({i * minibatch_size, 0, 0, 0}), mini_batch_shape)),
                    int(i)));
        }
        flowGraph.wait_for_all();
    }

    /**
     * backward function
     *
     * Performs backward propagation on the model using the given labels.
     *
     * @tparam T The data type of the Tensor
     * @tparam OutDim The output dimension of the model
     * @param labels The label Tensor
     * @param n_minibathes The number of minibatches
     */
    void backward(const Tensor<T, OutDim, Eigen::RowMajor> &labels, const int n_minibathes = 1) {
        size_t minibatch_size = labels.dimension(0) / n_minibathes;
        Eigen::array<size_t, 3> mini_batch_shape{minibatch_size, size_t(labels.dimension(1)), 1};
        for (size_t i = 0; i < n_minibathes; i++) {
            out_queue.try_pop(prediction);
            Tensor<T, OutDim, Eigen::RowMajor> grads = loss->calculate_grads(prediction.first, labels.slice(
                    Eigen::array<size_t, 3>({i * minibatch_size, 0, 0}), mini_batch_shape));
            modelOutNode.try_put(std::make_pair(std::move(grads), prediction.second));
        }
        flowGraph.wait_for_all();
    }

    /**
     * test function
     *
     * Tests the model on the given inputs and labels.
     *
     * @tparam T The data type of the Tensor
     * @tparam InpDim The input dimension of the model
     * @tparam OutDim The output dimension of the model
     * @param inputs The input Tensor
     * @param labels The label Tensor
     * @return A string representation of the test results
     */
    std::string test(const Tensor<T, InpDim, Eigen::RowMajor> &inputs, const Tensor<T, OutDim, Eigen::RowMajor> &labels) {
        modelTestNode.try_put(inputs);
        flowGraph.wait_for_all();
        std::stringstream result;
        result << "Loss: " << loss->calculate_loss(modelOutput, labels);
        double num_equal_examples = 0;
        for (int i = 0; i < modelOutput.dimension(0); ++i) {
            Tensor<bool, 0, Eigen::RowMajor> equal = ((modelOutput.chip(i, 0).argmax() == labels.chip(i, 0).argmax()));
            if (equal(0)) {
                num_equal_examples++;
            }
        }
        result << "  Accuracy: " << num_equal_examples << " / " << labels.dimension(0) << " : "
               << num_equal_examples / labels.dimension(0) * 100 << "%";
        return result.str();
    }

    /**
     * fit function
     *
     * Trains the model on the given inputs and labels for a specified number of epochs.
     *
     * @tparam T The data type of the Tensor
     * @tparam InpDim The input dimension of the model
     * @tparam OutDim The output dimension of the model
     * @param inputs The input Tensor
     * @param labels The label Tensor
     * @param epochs The number of training epochs
     * @param minibatch_size The size of each minibatch
     */
    void fit(const Tensor<T, InpDim, Eigen::RowMajor> &inputs, const Tensor<T, OutDim, Eigen::RowMajor> &labels, const size_t epochs, const int batch_size,
             const int n_minibathes) {
        Eigen::array<int, 4> batch_shape{batch_size, int(inputs.dimension(1)), int(inputs.dimension(2)), int(inputs.dimension(3))};
        Eigen::array<int, 3> batch_shape_y{batch_size, int(labels.dimension(1)), 1};
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            auto progress_bar = tq::tqdm(tq::range((int) inputs.dimension(0)/ batch_size));
            progress_bar.set_min_update_time(0.5);
            progress_bar.set_prefix("Epoch " + std::to_string(epoch));
            for (int i : progress_bar) {
                predict(inputs.slice(Eigen::array<int, 4>({i*batch_size, 0, 0, 0}), batch_shape), n_minibathes);
                backward(labels.slice(Eigen::array<int, 3>({i*batch_size, 0, 0}), batch_shape_y), n_minibathes);
            }
            std::cout << progress_bar.to_string() + " ";
            std::cout << test(inputs, labels) << std::endl;
        }
    }
};

template<class T, size_t Dim>
class SkipLayer{
    NodeTriplet<T, Dim> node1;
    NodeTriplet<T, Dim> node2;
public:
    /**
     * Constructor
     *
     * @param node1_ The first NodeTriplet to connect
     * @param node2_ The second NodeTriplet to connect
     */
    SkipLayer(NodeTriplet<T, Dim> &node1_, NodeTriplet<T, Dim> &node2_) : node1(node1_), node2(node2_) {}

    /**
     * Forward propagation
     *
     * Performs the forward propagation of the SkipLayer.
     * Adds the output of node1 to the input of the layer.
     *
     * @param inputs The input Tensor
     * @param minibatchInd The index of the minibatch (optional, default: 1)
     * @return The output Tensor after applying the forward propagation
     */
    Tensor<T, Dim+1, Eigen::RowMajor> forward(const Tensor<T, Dim+1, Eigen::RowMajor> & inputs, int minibatchInd = 1){
        return inputs + node1.get_minibatch(minibatchInd);
    }
};

/**
 * FlattenLayer Class
 *
 * This class represents a flatten layer in a neural network.
 * It reshapes the input data into a 2D tensor for further processing.
 *
 * Template Parameters:
 *   - T: The data type used for the computations (e.g., float, double)
 *   - Dim: The dimensionality of the input data (default: 3)
 *
 * Public Methods:
 *   - forward(...): Performs forward pass computation of the flatten layer.
 *   - backward(...): Performs backward pass computation of the flatten layer.
 */
template<class T>
class FlattenLayer{
private:
    int height;
    int width;
    int channels;
    int batch_size;
public:
    FlattenLayer() :
            height(0),
            width(0),
            channels(0),
            batch_size(0){}

    /**
    * Forward propagation
    *
    * Performs the forward propagation of the FlattenLayer.
    * Reshapes the input tensor to a 3D tensor.
    *
    * @param inputs The input Tensor
    * @param minibatchInd The index of the minibatch (optional, default: 1)
    * @param train Specifies if the forward propagation is performed during training (optional, default: false)
    * @return The output Tensor after applying the forward propagation
    */
    Tensor<T, 3, Eigen::RowMajor> forward(const Tensor<T, 4, Eigen::RowMajor> &inputs, int minibatchInd = 1, bool train = false) {
        batch_size = inputs.dimension(0);
        height = inputs.dimension(1);
        width = inputs.dimension(2);
        channels = inputs.dimension(3);
        return inputs.reshape(Eigen::array<int, 3>({static_cast<int>(inputs.dimension(0)), height * width * channels, 1}));
    }

    /**
     * Backward propagation
     *
     * Performs the backward propagation of the FlattenLayer.
     * Reshapes the output gradient tensor to match the input shape.
     *
     * @param out_gradient The output gradient Tensor
     * @param minibatchInd The index of the minibatch (optional, default: 1)
     * @return The input gradient Tensor after applying the backward propagation
     */
    Tensor<T, 4, Eigen::RowMajor>
    backward(const Tensor<T, 3, Eigen::RowMajor> &out_gradient, int minibatchInd=1) {
        return out_gradient.reshape(Eigen::array<int, 4>({batch_size, height, width, channels}));
    }

};

/**
 * Connect function template (NodeTriplet to NodeTriplet)
 *
 * Connects two NodeTriplets in a neural network by making the appropriate edges between their ports.
 *
 * @tparam T The data type of the Tensor
 * @tparam Dim1 The dimension of the first NodeTriplet
 * @tparam Dim2 The dimension of the second NodeTriplet
 * @param start The starting NodeTriplet
 * @param end The ending NodeTriplet
 */
template<class T, size_t Dim1, size_t Dim2>
void connect(NodeTriplet<T, Dim1> &start, NodeTriplet<T, Dim2> &end) {
    make_edge(start.forward, end.forward);
    make_edge(start.test, end.test);
    make_edge(end.backward, start.backward);
}

/**
 * Connect function template (SkipNode to NodeTriplet)
 *
 * Connects a SkipNode and a NodeTriplet in a neural network by making the appropriate edges between their ports.
 *
 * @tparam T The data type of the Tensor
 * @tparam Dim1 The dimension of the SkipNode
 * @tparam Dim2 The dimension of the NodeTriplet
 * @param start The starting SkipNode
 * @param end The ending NodeTriplet
 */
template<class T, size_t Dim1, size_t Dim2>
void connect(SkipNode<T, Dim1> &start, NodeTriplet<T, Dim2> &end){
    make_edge(start.forward, end.forward);
    make_edge(start.test, end.test);
    make_edge(end.backward, start.backward);
}

/**
 * Connect function template (NodeTriplet to FlattenNode)
 *
 * Connects a NodeTriplet and a FlattenNode in a neural network by making the appropriate edges between their ports.
 *
 * @tparam T The data type of the Tensor
 * @param start The starting NodeTriplet
 * @param end The ending FlattenNode
 */
template<class T>
void connect(NodeTriplet<T, 3> &start, FlattenNode<T> &end){
    make_edge(start.forward, end.forward);
    make_edge(start.test, end.test);
    make_edge(end.backward, start.backward);
}

/**
 * Connect function template (FlattenNode to NodeTriplet)
 *
 * Connects a FlattenNode and a NodeTriplet in a neural network by making the appropriate edges between their ports.
 *
 * @tparam T The data type of the Tensor
 * @param start The starting FlattenNode
 * @param end The ending NodeTriplet
 */
template<class T>
void connect(FlattenNode<T> &start, NodeTriplet<T, 2> &end){
    make_edge(start.forward, end.forward);
    make_edge(start.test, end.test);
    make_edge(end.backward, start.backward);
}

#endif //NEURALIB_MODEL_H