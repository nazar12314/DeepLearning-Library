#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>

class CudaMatrix {
private:
    bool device_allocated = false;
    bool host_allocated = false;

    void allocateCudaMemory();

    void allocateHostMemory();

public:
    std::pair<int, int> dims;

    std::shared_ptr<float> data_device;
    std::shared_ptr<float> data_host;

    CudaMatrix(size_t x_dim, size_t y_dim): dims({x_dim, y_dim}) {};

    CudaMatrix(size_t x_dim, size_t y_dim, float* data): dims({x_dim, y_dim}) {
        allocateMemory();
        std::move(data, data + dims.first * dims.second, data_host.get());
    }

    void allocateMemory();
    void allocateMemoryIfNotAllocated();

    void copyHostToDevice();
    void copyDeviceToHost();

    float& operator[](size_t index);
    const float& operator[](size_t index) const;
};