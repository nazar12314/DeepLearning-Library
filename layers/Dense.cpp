//
// Created by Naz on 3/31/2023.
//

#include "layers/Dense.h"
#include "layers/Layer.h"

//template <class T, size_t Dim>
//DenseLayer<T, Dim>::DenseLayer(const std::string &name, bool trainable, const Initializer &initializer):
//    Layer<T, Dim>(name, trainable), initializer(initializer) {}
//
//template<class T, size_t Dim>
//void DenseLayer<T, Dim>::forward(const Tensor<T, Dim> &) {
//
//}
//
//template<class T, size_t Dim>
//Tensor<T, Dim> DenseLayer<T, Dim>::backward(const Tensor<T, Dim> &) {
//    return Tensor<T, Dim>();
//}
//
//template<class T, size_t Dim>
//void DenseLayer<T, Dim>::set_weights(const Tensor<T, Dim> &) {
//
//}
//
//template<class T, size_t Dim>
//const Tensor<T, Dim> &DenseLayer<T, Dim>::get_weights() {
//    return Tensor<T, Dim>();
//}
//
//template<class T, size_t Dim>
//void DenseLayer<T, Dim>::adjust_weights(const Tensor<T, Dim> &) {
//
//};