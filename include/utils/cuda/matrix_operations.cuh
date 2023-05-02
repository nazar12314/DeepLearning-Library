#include <cuda.h>
#include <cuda_runtime.h>
#include "utils/cuda/CudaMatrix.cuh"

#define BLOCK_SIZE 32

namespace matrix_operations {
    void cuda_dense_backward(CudaMatrix& m1, CudaMatrix& m2, float *C_gpu, float batches);
}