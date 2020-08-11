#include "tensorflow/core/framework/op_kernel.h"

template <typename Device, typename T>
struct AdditionFunctor {
    void operator()(const Device& d, const T* a, const T* b, T* c, int N);
};

#if GOOGLE_CUDA
template <typename T>
struct AdditionFunctor<Eigen::GpuDevice,T> {
    void operator()(const Eigen::GpuDevice& d, const T* a, const T* b, T* c, int N);
};
#endif