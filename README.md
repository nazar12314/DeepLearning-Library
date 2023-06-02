# Parallel library of neural networks: Axon
Authors (team): Roman Bernikov, Nazar Demchuk, Nazar Kononenko, Liubomyr Oleksiuk <br>

## Prerequisites

Cmake, TBB, CUDA

## Compilation

```bash
./compile.sh
```

## Usage

### Create model
```c++
#include "Model.h"

int main(){
  Model<double, 4, 3> model("model", new optimizers::SGD<double>(0.05), new loss_functions::BinaryCrossEntropy<double>());

  auto input = model.addLayer<ConvolutionLayer<double>, 3>(28, 28, 1, 3, 2, "conv1", initializer);
  auto conv1 = model.addLayer<ConvolutionLayer<double>, 3>(26, 26, 2, 3, 2, "conv2", initializer);
  auto conv2 = model.addLayer<ConvolutionLayer<double>, 3>(24, 24, 2, 3, 2, "conv3", initializer);
  auto flatten = model.addFlattenLayer();
  auto dense1 = model.addLayer<DenseLayer<double>>(324, 20, "dense1", initializer);

  auto sigmoid1 = model.addLayer<activations::Sigmoid<double, 3>, 3>();
  auto sigmoid2 = model.addLayer<activations::Sigmoid<double, 3>, 3>();
  auto sigmoid3 = model.addLayer<activations::Sigmoid<double, 3>, 3>();
  auto out = model.addLayer<activations::Softmax<double, 2>>();

  connect(input, sigmoid1);
  connect(sigmoid1, conv1);
  connect(conv1, sigmoid2);
  connect(sigmoid2, conv2);
  connect(conv2, sigmoid3);
  connect(sigmoid3, flatten);
  connect(flatten, dense1);
  connect(dense1, out);
}
```

### Train model
```c++
model.setInput(input);
model.setOut(out);

model.fit(X_train, y_train, 10, 200, 4);
```
