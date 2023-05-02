#include "utils/cuda/CudaMatrix.cuh"

void CudaMatrix::copyDeviceToHost() {
    if (!host_allocated) {
        allocateHostMemory();
    }

    cudaMemcpy(data_host.get(), data_device.get(),
               dims.first * dims.second * sizeof(float),
               cudaMemcpyDeviceToHost);
}

void CudaMatrix::copyHostToDevice() {
    if (!device_allocated) {
        allocateCudaMemory();
    }

    cudaMemcpy(data_device.get(), data_host.get(),
               dims.first * dims.second * sizeof(float),
               cudaMemcpyHostToDevice);
}

void CudaMatrix::allocateMemoryIfNotAllocated() {
    if (!host_allocated || !device_allocated) {
        allocateMemory();
    }
}

void CudaMatrix::allocateMemory() {
    allocateHostMemory();
    allocateCudaMemory();
}

const float &CudaMatrix::operator[](size_t index) const {
    return data_host.get()[index];
}

float &CudaMatrix::operator[](size_t index) {
    return data_host.get()[index];
}

void CudaMatrix::allocateCudaMemory() {
    float* device_memory;
    cudaMalloc(&device_memory, dims.first * dims.second * sizeof(float));

    data_device = std::shared_ptr<float>(device_memory,
                                          [&](float* ptr){ cudaFree(ptr); });
    device_allocated = true;
}

void CudaMatrix::allocateHostMemory()  {
    data_host = std::shared_ptr<float>(new float[dims.first * dims.second],
                                        [&](const float* ptr){ delete[] ptr; });
    host_allocated = true;
}
