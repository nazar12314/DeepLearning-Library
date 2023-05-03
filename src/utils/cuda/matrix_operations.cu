#include "utils/cuda/matrix_operations.cuh"

__global__ void matrixMulKernel(double* A, double* B, double* C, int rowsA, int colsA, int colsB, double batches) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        double sum = 0;
        for (int k = 0; k < colsA; ++k) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum / batches;
    }
}

void matrix_operations::cuda_dense_backward(double *A, double *B, double *C_gpu, int rowsA, int colsA, int colsB, double batches) {
    double *A_dev, *B_dev, *C_dev;
    cudaMalloc(&A_dev, rowsA * colsA * sizeof(double));
    cudaMalloc(&B_dev, colsA * colsB * sizeof(double));
    cudaMalloc(&C_dev, rowsA * colsB * sizeof(double));
    cudaMemcpy(A_dev, A, rowsA * colsA * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B, colsA * colsB * sizeof(double), cudaMemcpyHostToDevice);

    int block_size = 32;
    dim3 dim_block(block_size, block_size);
    dim3 dim_grid((colsB + dim_block.x - 1) / dim_block.x, (rowsA + dim_block.y - 1) / dim_block.y);

    matrixMulKernel<<<dim_grid, dim_block>>>(A_dev, B_dev, C_dev, rowsA, colsA, colsB, batches);

    cudaMemcpy(C_gpu, C_dev, rowsA * colsB * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
}