#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "custom_kernel.h"
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <stddef.h>
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"



using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel
template <typename T>
__global__ void CudaAdditionKernel(const T* a, const T* b,T* c, int N){
    for(int idx: tensorflow::CudaGridRangeX(N)){
        c[idx] = a[idx] + b[idx];
    }
}

template <typename T>
void AdditionFunctor<GPUDevice, T>::operator()(const GPUDevice& d, const T* a, const T* b,
T* c, int N){
    int block_count = 1024;
    int thread_per_block = 20;
    CudaAdditionKernel<T> <<<block_count,thread_per_block,0,d.stream>>>(a,b,c,N);
    cudaError_t cudaErr = cudaDeviceSynchronize();
    if(cudaErr != cudaSuccess){
        printf("Failed to launch kernel with error of: \"%s\".\n",cudaGetErrorString(cudaErr));
    }
};

template struct AdditionFunctor<GPUDevice, float>;
template struct AdditionFunctor<GPUDevice, int>;


#endif // GOOGLE_CUDA