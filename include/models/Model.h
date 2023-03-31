//
// Created by Naz on 3/31/2023.
//

#ifndef NEURALIB_MODEL_H
#define NEURALIB_MODEL_H


#include <string>
#include <vector>
#include "layers/Layer.h"
#include "utils/Optimizer.h"
#include "utils/Loss.h"


template <class T, size_t Dim, class Func>
class Model {
    std::string name;
    std::vector<Layer<T, Dim>> layers;
    Optimizer<Func> optimizer;
    Loss loss;
};


#endif //NEURALIB_MODEL_H
