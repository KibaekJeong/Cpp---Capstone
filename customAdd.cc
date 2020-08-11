#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "custom_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

using namespace tensorflow;
using namespace shape_inference;

REGISTER_OP("CustomAddition")
    
    .Input("a: T")
    .Input("b: T")
    .Output("c: T")
    .Attr("T:{int32,int64,float32,float64}")
    .Doc(R"doc(Produce output by adding two Tensors)doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        ShapeHandle input;
        ShapeHandle output;
        //for each inputs
        for(size_t i = 0; i < c->num_inputs();++i){
            //validate input shape has shape of 4 dimension
            TF_RETURN_IF_ERROR(c->WithRank(c->input(i),4,&input));
            //use merge to validate shapes are all compatible
            TF_RETURN_IF_ERROR(c->Merge(output, input, &output));
        }
        
        c->set_output(0,output);
        return Status::OK();
    });



template<typename T>
struct AdditionFunctor<CPUDevice,T>{
    void operator()(const CPUDevice& d, const T* input_a, const T* input_b, T* output_c, int N){
        for(int i=0;i<N; i++){
            output_c[i] = input_a[i] + input_b[i];
        }
    }
};


template <typename Device, typename T>
class CustomAdditionOp: public OpKernel {
public:
    explicit CustomAdditionOp(OpKernelConstruction* context): OpKernel(context){
        //check attributes    
    
    }
    void Compute(OpKernelContext* context) override {
        //grab input tensor
        const Tensor& input_a_tensor = context->input(0);
        const Tensor& input_b_tensor = context->input(1);
        //flatten
        auto input_a = input_a_tensor.flat<T>();
        auto input_b = input_b_tensor.flat<T>();


        //create output tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context,context->allocate_output(0,input_a_tensor.shape(),&output_tensor));

        //create flatten version of output tensor to fill
        auto output_flat = output_tensor->flat<T>();

        const int N = output_flat.size();
        AdditionFunctor<Device, T>()(
            context->eigen_device<Device>(),
            input_a.data(),
            input_b.data(),
            output_flat.data(),
            N
        );
    }
};

// Register the CPU kernels.

#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CustomAddition").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CustomAdditionOp<CPUDevice, T>)

REGISTER_CPU(int);
REGISTER_CPU(float);


// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CustomAddition").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CustomAdditionOp<GPUDevice, T>)
      
REGISTER_GPU(int);
REGISTER_GPU(float);
#endif 


