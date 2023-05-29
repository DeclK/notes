# Pytorch C++ & CUDA Extension

这里想要以 BEVDet & Deformabel Attention 作为学习的样例，了解 Pytorch C++ & CUDA extension 的原理

有几个问题需要解决：

1. 如何创建普通的 C++ & CUDA extension
2. C++ & CUDA extension 之间有没有什么区别与联系
3. 在什么场合之下可以使用其加速：
   1. 可以将一些运算进行融合的操作（不太理解融合
   2. 可以将运算并行？也许是一种融合
4. 使用 C++ & Pytorch 需要什么特别的语法或者规则

## Get Started

reference: [zhihu](https://zhuanlan.zhihu.com/p/100459760) [pytorch tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension)

把大象放进冰箱需要3步，写 C++ Extension 也需要3步

1. 用 C++ 写好自定义层，包括前向传播和反向传播
2. 用 setuptools 编译并生成 C++ 拓展
3. 使用 pytorch Function 封装拓展，然后就可以按照正常模块使用了

先看看最简单的例子，写一个 `z = x + 2y`

pybind 是将编译好的 C++ 算子暴露给 python 接口

start from a minimum CUDA example! 

用 gpt4 生成了一个最小例子，先创建 `my_extension.cpp`

```cpp
#include <torch/extension.h>

void my_cuda_kernel(torch::Tensor tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_cuda_kernel", &my_cuda_kernel, "My CUDA kernel");
}
```

再创建 `my_extension.cu`

```cpp
#include <torch/extension.h>

__global__ void my_kernel(float* data, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        data[index] = index;
    }
}

void my_cuda_kernel(torch::Tensor tensor) {
    const int size = tensor.size(0);
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;

    float* data = tensor.data<float>();

    my_kernel<<<blocks, threads>>>(data, size);

    cudaDeviceSynchronize();
}
```



.cu 文件名不能和 .cpp 文件名同名！

<torch/torch.h> 就相当于是 C++ 版本的 torch，而 <torch/extension.h> 相当于是专门为 c++ extension 所定义的接口，所以前者更加强大一些

[bilibili](https://www.bilibili.com/video/BV1pG411F7Yx)

## Grammar

1. 可以使用 Pytorch C++ version 来直接编写 Pytorch C++ 版本 [bilibili](https://www.bilibili.com/video/BV1vF411J7fW)
2. `::` 代表着什么，有两个用途：namespace & class member access
3. `g++ & cmake` 是什么？有什么用？cmake 能够编译一整个项目，能够自动寻找文件之中的引用和依赖，进行逐个编译，省去了手动编译，g++ & gcc 都是 C++ 编译器