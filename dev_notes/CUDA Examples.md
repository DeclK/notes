## Learn Pytorch CUDA Development with Examples

通过学习一些例子来学习 pytorch CUDA 开发，学习目的：

- [x] 熟悉 C++/CUDA 编程
- [x] 如何构建和使用 extension
- [ ] 如何构建 **Plugins**
- [ ] 如何使用 Cmake

## Trilinear Interpolation

根据 [bilibili](https://www.bilibili.com/video/BV1pG411F7Yx/) 进行整理，对应 [github](https://github.com/kwea123/pytorch-cppcuda-tutorial)，参考 [doc](https://pytorch.org/tutorials/advanced/cpp_extension.html)

我先从程序的运行过程，对整个流程进行整理，不然会失去对逻辑的整体把握。然后再对其中的知识点进行补充介绍

### pytorch-cpp-cuda 流程

1. 程序会从 Pytorch 接口来调用算子，这里需要注意：所有的输入最好使用 `.contigous()` 来保证其存储在屋里层面上是连续的

   ```python
   class Trilinear_interpolation_cuda(torch.autograd.Function):
       @staticmethod
       def forward(ctx, feats, points):
           feat_interp = cppcuda_tutorial.trilinear_interpolation_fw(feats, points)
           ctx.save_for_backward(feats, points)
           return feat_interp
   
       @staticmethod
       def backward(ctx, dL_dfeat_interp):
           feats, points = ctx.saved_tensors
           dL_dfeats = cppcuda_tutorial.trilinear_interpolation_bw(dL_dfeat_interp.contiguous(), feats, points)
           return dL_dfeats, None
   ```

2. 然后通过 cpp 作为 pytorch & CUDA 之间的桥梁

   ```cpp
   #include "utils.h"
   #include <iostream>
   
   #define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
   #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
   #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
   
   torch::Tensor trilinear_interpolation_fw(
       const torch::Tensor feats, const torch::Tensor points
   ){
       CHECK_INPUT(feats); // CHECK every inptus...
       CHECK_INPUT(points);
       return trilinear_fw_cu(feats, points);
   }
   
   torch::Tensor trilinear_interpolation_bw(
       const torch::Tensor dL_dfeat_interp, const torch::Tensor feats, const torch::Tensor points
   ){//...
   }
   
   // Bind funtions to extension module
   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
       m.def("trilinear_interpolation_fw", &trilinear_interpolation_fw);
       m.def("trilinear_interpolation_bw", &trilinear_interpolation_bw);
   }
   
   ```

3. 最后实现 CUDA 算子

   ```cpp
   // take forward as example
   torch::Tensor trilinear_fw_cu(
       const torch::Tensor feats, const torch::Tensor points
   ){
       const int N = feats.size(0), F = feats.size(2);
       
       torch::Tensor feat_interp = torch::empty({N, F}, feats.options());
   
       const dim3 threads(16, 16);
       const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);
   
       AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu", 
       ([&] {
           trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
               feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
               points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
               feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
           );
       }));
   
       return feat_interp;
   }
   ```

### Dispatch & PackedTensorAccessor

重点关注了 `interpolation_kernel.cu` 中的代码，整理一下不太熟悉的地方。code snippet 请参考上方的 `trilinear_fu_cu`

1. `AT_DISPATCH_FLOATING_TYPES` macro

   - 该 macro 是用于处理多种形式的浮点数据的，例如 `float & double` 都可以统一处理，使得代码更简洁，本质是如下代码

     ```c++
     switch (tensor.type().scalarType()) {
       case torch::ScalarType::Double:
         return function<double>(tensor.data<double>());
       case torch::ScalarType::Float:
         return function<float>(tensor.data<float>());
       ...
     }
     ```

   - macro 中的参数 `scalar_t` 就是 `feats.type()`，在代码中作为模板

2. `Tensor.packed_accessor<scalar_t, N, torch::RestrictPrtTraits, size_t>`

   - `size_t` 是将就是 `int32`，是 C++ 中标准库的定义 `typedef unsigned long size_t`，代表索引 index 所使用的数据类型。使用 `int32` 会比 `int64` 索引更快
   - `N` 代表的是该 tensor 维度的数量，例如 `shape = (N,H,W,C)` 的维度数量为 4

   - `torch::RestrictPtrTraits` 表示使用 `__restrict__` 修饰指针，即其指向的对象不会被别的指针所引用，等价于 `scalar_t* __restrict__ xxx`，这样编译出来的程序会更快些

   - 可以使用 `Tensor.packed_accessor32 or Tensor.packed_accessor64`，这样就不用再填入参数 `size_t`，只需要三个参数 `<scalar_t, N, torch::RestrictPtrTraits>`

   - 使用了额 `packed_accessor` 过后，就可以**直接使用多维索引获取元素，而不需要计算一维下的指针索引**，这样方便于代码的书写与可读性，下面是 pytorch doc 中的代码片段，其使用的是原始指针，所以需要用一维索引来获取元素

     ```c++
     template <typename scalar_t>
     __global__ void lltm_cuda_forward_kernel(
         const scalar_t* __restrict__ gates,
         const scalar_t* __restrict__ old_cell,
         scalar_t* __restrict__ new_h,
         scalar_t* __restrict__ new_cell,
         scalar_t* __restrict__ input_gate,
         scalar_t* __restrict__ output_gate,
         scalar_t* __restrict__ candidate_cell,
         size_t state_size) {
       const int column = blockIdx.x * blockDim.x + threadIdx.x;
       const int index = blockIdx.y * state_size + column;
       const int gates_row = blockIdx.y * (state_size * 3);
       if (column < state_size) {
         input_gate[index] = sigmoid(gates[gates_row + column]);
         output_gate[index] = sigmoid(gates[gates_row + state_size + column]);
         candidate_cell[index] = elu(gates[gates_row + 2 * state_size + column]);
         new_cell[index] =
             old_cell[index] + candidate_cell[index] * input_gate[index];
         new_h[index] = tanh(new_cell[index]) * output_gate[index];
       }
     }
     ```

     实测使用 `PackedTensorAccessor32` 肯定是有延时的，在 A10 上 使用输入 `(16, 32, 128)` 的张量，会多 ~250 ns，几乎忽略不计

### Back Propagation

为了让算子能够加入到训练过程中，不仅要实现前向传播的方法，也要实现反向传播。计算反向传播，即需要计算所有 output 对所有 trainable input 的偏微分，最后依然要将算子使用 `torch.autograd.Function` 中，才能够在 torch 中享受自动微分的好处 。判断 input 是否需要计算梯度的方法：模型权重的改变是否影响该 input。并且我们还可以借助一些网站来帮助写偏微分表达式：

1. [symbolab](https://www.symbolab.com/solver/derivative-calculator)
2. [derivative-calculator](https://www.derivative-calculator.net/)

实现 CUDA 的反向传播后需要与 pytorch 实现进行对比，保证实现的正确性

## BEV Pool

能不能将BEVPool 改成 accessor，理解 BEVDet 的原理，如何添加 BEVPool Plugin

## CMake

使用了 cmake 来测试 libtorch，可能需要了解下 shell 编程
