//
// Created by Naz on 3/31/2023.
//

#include "layers/Dense.h"
#include "layers/Layer.h"

template <class T, size_t Dim>
Dense<T, Dim>::Dense(const std::string &name, bool trainable, const Initializer &initializer):
    Layer<T, Dim>(name, trainable), initializer(initializer) {}