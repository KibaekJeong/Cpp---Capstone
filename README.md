# CppND-Capstone (Tensorflow Custom Op) Project
---

In this project we will create a custom op that isn't covered by existing Tensorflow Library. Following project is created by following instructions provided by [Tensorflow](https://www.tensorflow.org/), in the [Create an op](https://www.tensorflow.org/guide/create_op) page.

For further instruction, please refer to [Create an op](https://www.tensorflow.org/guide/create_op) page.

---
### Tensor Addition Op
Addition op is an op that gets two tensors as input and outputs a tensor that is summation of input tensors.

Following is the Functor of Addition Op
```
template<typename T>
struct AdditionFunctor<CPUDevice,T>{
    void operator()(const CPUDevice& d, const T* input_a, const T* input_b, T* output_c, int N){
        for(int i=0;i<N; i++){
            output_c[i] = input_a[i] + input_b[i];
        }
    }
};
```
Output tensor c is summation of input tensor a and b.

---
### Project structure
- customAdd.cc

  &#8594; code of custom Addition op
- custom_kernel.cu.cc

  &#8594; Specialization for the GPU device defined
- custom_kernel.h

  &#8594; header file for customAdd op
- customAdd.so

  &#8594; Shared library created after customAdd.cc is built
- customAdd_test.py

  &#8594; File for checking whether customAdd op is working  properly


---
### Building the op Library
First of all, using python, we will get the header directory and the get_lib directory.
```
$ python
>>> import tensorflow as tf
>>> tf.sysconfig.get_include()
>>> tf.sysconfig.get_lib()
```

### Compile
#### Compile with CPU Device
Run following codes to compile custom op into a dynamic library.
```
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```

* Note on gcc version >=5: gcc uses the new C++ ABI since version 5. The binary pip packages available on the TensorFlow website are built with gcc4 that uses the older ABI. If you compile your op library with gcc>=5, add -D_GLIBCXX_USE_CXX11_ABI=0 to the command line to make the library compatible with the older abi.

#### Compile with GPU Device
Using CUDA kernel to implement op.
```
nvcc -std=c++11 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o cuda_op_kernel.so cuda_op_kernel.cc \
  cuda_op_kernel.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}
```

### Testing through python
Compile with testing file in python code.
```
python customAdd_test.py
```
When operation is successfully done, you will see
```
Operation Successful!
```
---
### Testing Result

![output](./image/output.png)
---
### Prerequisites
- [Tensorflow binary](https://www.tensorflow.org/install)
- g++
- CUDA if running with GPU Device
---
### Project Specification
#### README
- A README with instructions is included with the project
- The README indicates which project is chosen.
- The README includes information about each rubric point addressed.
#### Compiling and Testing
- The submission must compile and run.
#### Loops, Functions, I/O
- The project demonstrates an understanding of C++ functions and control structures.
- The project reads data from a file and process the data, or the program writes data to a file.
#### Object Oriented Programming
- The project uses Object Oriented Programming techniques.
- Classes abstract implementation details from their interfaces.
- Classes encapsulate behavior.
- Classes follow an appropriate inheritance hierarchy.
- Overloaded functions allow the same function to operate on different parameters.
- Derived class functions override virtual base class functions.
- Templates generalize functions in the project.
