#include <iostream>
#include "utils/cuda/matrix_operations.cuh"

__global__ void dense_backward_kernel(float* A, float* B, float* C, int rowsA, int colsA, int colsB, float batches) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float sum = 0;
        for (int k = 0; k < colsA; ++k) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum / batches;
    }
}

void matrix_operations::cuda_dense_backward(CudaMatrix& A, CudaMatrix& B, float *C_gpu, float batches) {
    float *C_dev;

    int rowsA = A.dims.first, colsA = A.dims.second, colsB = B.dims.second;

    cudaMalloc(&C_dev, rowsA * colsB * sizeof(float));

    A.copyHostToDevice();
    B.copyHostToDevice();

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((colsB + dim_block.x - 1) / dim_block.x, (rowsA + dim_block.y - 1) / dim_block.y);

    dense_backward_kernel<<<dim_grid, dim_block>>>(A.data_device.get(), B.data_device.get(), C_dev, rowsA, colsA, colsB, batches);

    cudaMemcpy(C_gpu, C_dev, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Multiplication completed successfully!" << std::endl;

    cudaFree(C_dev);
}
