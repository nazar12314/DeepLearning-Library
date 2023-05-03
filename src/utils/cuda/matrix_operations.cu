#include "utils/cuda/matrix_operations.cuh"

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int rowsA, int colsA, int colsB, double batches) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        double sum = 0;
        for (int k = 0; k < colsA; ++k) {
            sum += A[row * colsA + k] * B[col * colsA + k];
        }
        C[row * colsB + col] = sum / batches;
    }
}

void matrix_operations::cuda_dense_backward(const double *A, const double *B, double *C_gpu, int rowsA, int colsA, int colsB, double batches) {
    double *A_dev, *B_dev, *C_dev;
    cudaMalloc(&A_dev, rowsA * colsA * sizeof(double));
    cudaMalloc(&B_dev, colsA * colsB * sizeof(double));
    cudaMalloc(&C_dev, rowsA * colsB * sizeof(double));
    cudaMemcpy(A_dev, A, rowsA * colsA * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B, colsA * colsB * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((colsB + dim_block.x - 1) / dim_block.x, (rowsA + dim_block.y - 1) / dim_block.y);

    matrixMulKernel<<<dim_grid, dim_block>>>(A_dev, B_dev, C_dev, rowsA, colsA, colsB, batches);

    cudaMemcpy(C_gpu, C_dev, rowsA * colsB * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
}

__global__ void forward_kernel(const double* A, const double* B, double* C, int rowsA, int colsA, int colsB, const double* biases) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        double sum = 0;
        for (int k = 0; k < colsA; ++k) {
            sum += A[row * colsA + k] * B[col * colsA + k];
        }
        C[row * colsB + col] = sum + biases[col];
    }
}

void matrix_operations::cuda_dense_forward(double *A, double *B, double *C_gpu, int rowsA, int colsA, int colsB, double* biases) {
    double *A_dev, *B_dev, *C_dev, *biases_dev;
    cudaMalloc(&A_dev, rowsA * colsA * sizeof(double));
    cudaMalloc(&B_dev, colsA * colsB * sizeof(double));
    cudaMalloc(&C_dev, rowsA * colsB * sizeof(double));
    cudaMalloc(&biases_dev, colsB * sizeof(double));

    cudaMemcpy(A_dev, A, rowsA * colsA * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B, colsA * colsB * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(biases_dev, biases, colsB * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((colsB + dim_block.x - 1) / dim_block.x, (rowsA + dim_block.y - 1) / dim_block.y);

    forward_kernel<<<dim_grid, dim_block>>>(A_dev, B_dev, C_dev, rowsA, colsA, colsB, biases_dev);

    cudaMemcpy(C_gpu, C_dev, rowsA * colsB * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    cudaFree(biases_dev);
}