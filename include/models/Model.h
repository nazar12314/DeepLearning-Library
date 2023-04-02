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


template <class T, class Func>
class Model {
    std::string name;
    std::vector<Layer<T>> layers;
    Optimizer<T, Func> optimizer;
    Loss loss;
};


#endif //NEURALIB_MODEL_H
