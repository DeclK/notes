# Pytorch Deployment

考虑使用 ONNX 相关工具链进行部署，参考 [onnx tutorial](https://github.com/onnx/tutorials)	[onnx runtime](https://onnxruntime.ai/docs/)

[zhihu mmlab 部署教程](https://zhuanlan.zhihu.com/p/477743341)

[zhihu mmlab 部署教程 四](https://zhuanlan.zhihu.com/p/513387413)

[pytorch tutorial c++ extension](https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension)

[pytorch c++ tutorial github](https://github.com/prabhuomkar/pytorch-cpp)

[pytorch c++ extension 知乎](https://zhuanlan.zhihu.com/p/100459760)

## C++ extension

先说一下基本的流程：

- 利用 C++ 写好自定义层发功能，主要包括前向传播和方向传播，以及 pybind11 的内容
- 写好 setup.py 脚本， 并利用 python 提供的 setuptools 来编译并加载 C++ 代码
- 编译安装，在 python 中调用 C++ 扩展接口



第一步，编写头文件 `test.h`

这里包含一个重要的头文件 `<torch/extension.h>`

这个头文件里面包含很多重要的模块。如 pybind11，以及包含 Tensor 的一系列定义操作

`#ifdef WITH_CUDA` 什么意思？

`#pragma once`

第二步，写源文件 `test.cpp`

源文件cpp里面包含了三个部分，第一个部分是forward函数，第二个部分是backward函数，第三个部分是pytorch和C++交互的部分

```c++
/*test.cpp*/
#include "test.h"

// part1:forward propagation
torch::Tensor Test_forward_cpu(const torch::Tensor& x, const torch::Tensor& y)
{
    AT_ASSERTM(x.sizes() == y.sizes());
    torch::Tensor z = torch::zeros(x.sizes());
    z = 2 * x + y;
    return z;
}

//part2:backward propagation
std::vector<torch::Tensor> Test_backward_cpu(const torch::Tensor& gradOutput)
{
    torch::Tensor gradOutputX = 2 * gradOutput * torch::ones(gradOutput.sizes());
    torch::Tensor gradOutputY = gradOutput * torch::ones(gradOutput.sizes());
    return {gradOutputX, gradOutputY};
}

// part3:pybind11 （将python与C++11进行绑定， 注意这里的forward，backward名称就是后来在python中可以引用的方法名）
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &Test_forward_cpu, "Test forward");
    m.def("backward", &Test_backward_cpu, "Test backward");
}
```



第三步，编写 `setup.py`

```python
from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension

# 头文件目录
include_dirs = os.path.dirname(os.path.abspath(__file__))
#源代码目录 
source_file = glob.glob(os.path.join(working_dirs, 'src', '*.cpp'))

setup(
    name='test_cpp',  # 模块名称
    ext_modules=[CppExtension('test_cpp', sources=source_file, include_dirs=[include_dirs])],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```



最后一步，封装 extension

