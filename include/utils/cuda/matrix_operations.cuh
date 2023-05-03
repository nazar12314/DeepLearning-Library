#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace matrix_operations {
    void cuda_dense_backward(double *A, double *B, double *C_gpu, int rowsA, int colsA, int colsB, double batches);
}