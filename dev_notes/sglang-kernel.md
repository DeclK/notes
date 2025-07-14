# SGLang Kernel

一开始我还在纠结看 sglang 好，还是看 vllm 好，最后发现两个都不好上手。以下是我还在纠结时的笔记：

综合看下来，sglang 的代码其实比 vllm 写得更加简单，虽然大家都在说 vllm 比 sglang 好上手，但从代码量来看并非如此

```txt
👉SGLANG👈
--------------------------------------------------------------------------------
Language                      files          blank        comment           code
--------------------------------------------------------------------------------
CUDA                             36           1107            927           8801
C++                              20           1062            814           7247

👉VLLM👈
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
CUDA                           112           5072           3888          30277
C/C++ Header                    44           1515           1047           7752
C++                             13            424            305           2618
```

可以看到 vllm 的代码数量几乎是 sglang 的三倍。这是因为 sglang 中还借用了 vllm 的代码，例如 sglang 里就没有提供 w4a16 marlin kernel 的实现，而是直接复用了 vllm 中的 layer，或许我们可以自己集成到 sglang 当中🤔

sglang 额外给 sm80 写了 cutlass exetension，看上去是把 cutlass 2.x 的两个文件给搬过来了。看来我势必得把 cutlass 2.x 和 cutlass 3.x 分开来用才行，显然 3.x 对 2.x 的兼容并不好

我本来希望完成的学习目标：

1. 如何使用 cutlass api 完成所需要的矩阵乘法
2. 如何使用 epilogue 完成算子融合

3. 如何高效地构建 profile/benchmark 脚本

   - 如何在 python 中构建测试脚本
   - 如何将 cuda kernel 绑定到 torch
     - 如何构建 `CMakelists.txt & setup.py`
     - 如何将项目进行打包，形成一个 wheel 文件

中途还发现了 [gemm-int8](https://github.com/IST-DASLab/gemm-int8) 项目，本来想深入学习这个小项目的，发现这个项目在 hopper 上性能很差，几乎跟 fp16 一样，所以必须使用 cutlass 3.x 接口来加速

最后经过一番思考还是得出结论：cutlass 不适合学习，只适合使用。基于这个结论，我其实要做的就是学习那些 sglang 是如何使用 cutlass 的，我只需要借用其代码，作为我的算子“代理”即可。如果我真的要深入学，我估计会看 svdquant 中的推理引擎框架，其有自己的 gemm struct，将 mainloop & epilogue 只做了简单的抽象，没有 cutlass 复杂的模板，并且包含各种融合算子。另外再看 svdquant 中发现 Hopper GPU 不支持 4-bit tensor core [issue](https://github.com/mit-han-lab/nunchaku/issues/268)

总结：学习目标就大缩水，核心变为了如何构建 cuda cpp python 项目，以 sglang-kernel 为例

## How to debug CUDA

在观看 [bilibili](https://www.bilibili.com/video/BV1kToTY6Eh5?spm_id_from=333.788.videopod.episodes&p=8) 的时候发现了 CUDA 其实是可以进行 debug 的，但是在自己实际操作的时候发现，是真的不好用。不过还是将经验总结一下，毕竟花了一整天看这个😅参考资料 [zhihu](https://zhuanlan.zhihu.com/p/508810115) [blog](https://fancyerii.github.io/2024/01/17/vscode-cuda-debug/) [github issue](https://github.com/graphdeco-inria/gaussian-splatting/issues/827)

对于 vscode 来说，只需要安装两个插件 c++ & nsight system 即可，然后通过 `launch.json` 配置需要运行的 executable 即可。而获得 executable 需要有 debug 信息才能正常进入断点。此时需要再编译的时候加入 `-g -G` flag 以加入 debug 信息，其中大小 g 分别代表 host code & device code

如果直接使用 `nvcc` 来进行编译，那么直接加 `-g -G` 即可。但通常项目构建会使用 cmake，所以需要用 cmake 命令

```cmake
target_compile_options(your_target PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
```

当进行配置过后的确能很容易进入 host code debug 断点，但是对于 cuda code debug 断点表现得相当奇怪，以下是一些简要现象：

1. 无法通过 step in 进入 cuda code，必须在 cuda code 里打下断点，才能在停在 cuda code 中的断点里。换句话说，如果 cuda code 里没有断点，是无法进入 cuda code debug 的
2. 可以直接使用 `cuda-gdb .program` 加 `step` 命令来查看代码运行位置
3. 加入了 debug flag 过后，真实运行程序可能会表现不一样，包括但不限于：程序出错、程序卡死等
4. **print 才是 cuda 最终的 debug 方法！**

## .cu .cuh .h .cpp .cc

头文件类：`.cuh & .h`，这一类文件是不会出现在 `add_library or add_executable` 当中的，而且一般不会将实现放在其中，除非为了效率考量，会将 inline function 写在头文件当中。在头文件中通常会写入三类代码：

1. function/class declaration
2. macro helper
3. inline function
4. template function/struct

源文件类：`.cu & .cc & .cpp`，这三个是 cuda & c++ 源文件，是 cooking 的“原材料”，会实际地进行编译！

在 sglang kernel 中代码实现按照如下

```txt
sglang/sgl-kernel/csrc
├── allreduce
│   ├── custom_all_reduce.cu
│   ├── custom_all_reduce.cuh
|	...
├── attention
│   ├── cascade.cu
│   ├── cutlass_mla_kernel.cu
|	...
├── cpu
│   ├── CMakeLists.txt
│   ├── activation.cpp
|	...
├── cutlass_extensions
│   ├── epilogue
│   │   └── epilogue_per_row_per_col_scale.h
│   └── gemm
│       ├── gemm_universal_base_compat.h
│       └── gemm_with_epilogue_visitor.h
├── elementwise
│   ├── activation.cu
│   ├── fused_add_rms_norm_kernel.cu
│   └── rope.cu
├── gemm
│   ├── awq_kernel.cu
│   ├── bmm_fp8.cu
|	...
├── moe
│   ├── cutlass_moe_helper.cu
│   ├── fp8_blockwise_moe_kernel.cu
|	...
├── common_extension.cc
├── flash_extension.cc
└── torch_extension_rocm.cc
```

`csrc` 包含了所有的 `.cu` 文件（kernel 实现），然后通过 `.cc` 文件 binding 到 pytorch 当中。其中 `.cc` 文件中的 kernel 全部由头文件声明引入，而头文件单独放在 `csrc` 之外的 `include` 文件夹当中

```txt
sglang/sgl-kernel/include
├── sgl_flash_kernel_ops.h
├── sgl_kernel_ops.h
├── sgl_kernel_torch_shim.h
└── utils.h
```

最终将所有的 `.cc & .cu` 源文件添加到 library 当中，完成编译

```cmake
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/csrc
    ${repo-cutlass_SOURCE_DIR}/include
    ${repo-cutlass_SOURCE_DIR}/tools/util/include
    ${repo-flashinfer_SOURCE_DIR}/include
    ${repo-flashinfer_SOURCE_DIR}/csrc
)
set(SOURCES
    "csrc/allreduce/custom_all_reduce.cu"
    "csrc/attention/cascade.cu"
    ...
    "csrc/common_extension.cc"
)

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})
```

其中将 `csrc` 文件夹也添加到 include 路径当中，是因为其中也有一些子目录中包含头文件（例如 `cutlass_extention`），通过多级路径进行 include `#include "cutlass_extentions/..."`

## kernel launch and dispatch wrapper

一般在 pytorch 中调用 kernel 的逻辑有四层，我自底向上进行整理

1. global kernel function，核心的 CUDA kernel 代码，以 `__global__` 进行标识
2. launching function，为 host function，用于 laucn kernel，需要填入 launch 参数：grid，block，stream，shared memory。这一步一般在 cutlass 中已经被包装好了，调用的是 `gemm_op(·)`
3. torch function，将 launch function 包装起来，使得其能够接受输入参数为 torch tensor。通过传入 torch tensor 的相关信息（tensor shape，data pointer）给 launch function，从而运行 GPU。此时由于 cutlass 的原因，torch function 还可能是一个模板函数，这些模板参数会被 cutlass 使用到。那么为了将 torch template function 进行实例化，则诞生了下一层的 dispatch 逻辑
4. dispatch function，需要确定 gemm 模板参数，包括：kernel shape, output dtype, sm version 等，此时就需要 dispatch function 来细化。对于不同 shape 的输入 tensor，可以选择不同 kernel shape 来获得更优性能。同时在 dispatch 之前还会对输入进行一些 check，以给出报错信息

其实如果没有 cutlass 模板的话， 第二层和第三层的调用逻辑就可以合并起来，使得整个结构变得简单

## cmake command

对 cmake 中的命令做一些整理，并给出一个一般的构建 cuda extention 流程

1. Compile options

   sglang 首先定义了一些基础 nvcc flags，然后根据 cuda version 或者 enable option 再对 flag 调整。这里我对一些常见的 flags 做整理

   ```cmake
   set(SGL_KERNEL_CUDA_FLAGS
       "-DNDEBUG"	# Defines NDEBUG macro
       "-DOPERATOR_NAMESPACE=sgl-kernel"
       "-O3"		# O3, is the highest optimization level
       "-gencode=arch=compute_75,code=sm_75"	# Generates code for different NVIDIA GPUs
       "-gencode=arch=compute_80,code=sm_80"
       "-gencode=arch=compute_89,code=sm_89"
       "-gencode=arch=compute_90,code=sm_90"
       "-std=c++17"
       "--expt-relaxed-constexpr"	# Allow host & device code to invoke __device__ & __host__ constexpr functions
       "--expt-extended-lambda"	# Allow __host__, __device__ annotations in lambda declaration
       "--threads=32"				# Threads for compile
   
       # Suppress warnings
       "-Xcompiler=-Wconversion"
       "-Xcompiler=-fno-strict-aliasing"
       # "-use_fast_math" # Fast math method for older CUDA versions
   )
   ```

2. find packages

   我们需要使用 python & torch package 来构建 torch extension，当然 cuda package 肯定也是需要的

   ```cmake
   # Python
   find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)
   # SKBUILD_SABI_COMPONENT is automatically introduced by python cmake config (if has it)
   
   # CUDA
   find_package(CUDAToolkit REQUIRED)
   set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)
   # implicitly enbale separate compilation for cuda, might not be necessary but recommend
   
   # Torch
   find_package(Torch REQUIRED)
   ```

   有时候我在一个 venv 当中找不到 torch，可能是因为我的 torch 是从 system-packages 当中导入，自然不在 venv 中的 site-packages 当中，所以需要将 torch 的 cmake config 路径传入

   ```cmake
   list(APPEND CMAKE_PREFIX_PATH "/usr/local/lib/python3.10/dist-packages/torch/share/cmake")
   ```

3. python add library

   在之前编译库都是直接用 `add_library`，但是现在要编译的库需要能够在 python 当中通过 `import` 进行加载 (load)，所以需要使用 `Python_add_library` 命令。在 add library 过后则需要指明 include 路径和额外所需的 library：pytorch & cuda

   ```cmake
   Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})
   # the ABI part can be treated as fixed tempalte in python binding
   target_include_directories(common_ops PRIVATE ...)
   target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt)
   ```

4. Fetch content

   FetchContent 可用于下载和集成外部项目，一般用法如下

   > - `FetchContent_Declare`: Specifies the repository details:
   >   - `GIT_REPOSITORY:` The URL (e.g., https://github.com/NVIDIA/cutlass).
   >   - `GIT_TAG`: The commit or tag to use (e.g., f115c3f8... for cutlass).
   > - `FetchContent_Populate`: Downloads the repository into the build directory (e.g., `_deps/repo-cutlass-src`).

   通过 declare 当中的 name 来获得该项目的源文件路径，like `repo-flashinfer_SOURCE_DIR`

5. ccache

   `ccache`（Compiler Cache）是一个编译器缓存工具，主要用于 **加速重复编译**。我也当做固定流程来理解好了

   ```cmake
   # ccache option
   option(ENABLE_CCACHE "Whether to use ccache" ON)
   find_program(CCACHE_FOUND ccache)
   if(CCACHE_FOUND AND ENABLE_CCACHE AND DEFINED ENV{CCACHE_DIR})	# env var must have CCACHE_DIR
       message(STATUS "Building with CCACHE enabled")
       set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "ccache")
       set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "ccache")
   endif()
   ```

   在启用 ccache 时，需要指定一些环境变量

   ```shell
   apt-get install -y ccache
   # Building with ccache is enabled when ccache is installed and CCACHE_DIR is set.
   export CCACHE_DIR=/path/to/your/ccache/dir
   export CCACHE_BACKEND=""
   export CCACHE_KEEP_LOCAL_STORAGE="TRUE"
   unset CCACHE_READONLY
   python -m uv build --wheel -Cbuild-dir=build --color=always .
   ```

6. install

   在 CMake 中，`install()` 命令用于定义项目安装规则，指定构建产物（可执行文件、库、头文件等）在系统上的最终安装位置。当用户执行 `make install`（或等价命令）时，这些文件会被复制到指定路径

   ```cmake
   install(TARGETS common_ops LIBRARY DESTINATION sgl_kernel)
   ```

   `common_ops.so` 会被移动到目录 `sgl_kernel` 下，其前缀目录决定于 `CMAKE_INSTALL_PREFIX`。如果使用 wheel 来打包的话，目录会变成 wheel 打包文件夹目录，具体情况可见下一个小节

7. together with `pyproject.toml`

   ```toml
   [build-system]
   requires = [
     "scikit-build-core>=0.10",
     "torch>=2.6.0",
     "wheel",
   ]
   build-backend = "scikit_build_core.build"
   
   [project]
   dependencies = []
   
   [tool.wheel]
   exclude = [
     "dist*",
     "tests*",
   ]
   
   [tool.scikit-build]
   cmake.build-type = "Release"
   minimum-version = "build-system.requires"
   
   wheel.py-api = "cp39"
   wheel.license-files = []
   wheel.packages = ["python/sgl_kernel"]
   ```

   将所编译的 `.so` 以及其他文件全部打包到一个 `.whl` 文件当中，然后通过 `pip install *.whl` 完成下载，其中需要配置的只有 `wheel.packages`，这里包含了想要包含的 package 路径，其中必须要有 `__init__.py` 文件。同时在打包的时候，会将 cmake 当中 `install` 文件一块打包，并且会将同名的依赖进行合并（例如 `common_ops.so` 合并到 `sgl_kernel` 目录之下），其结构目录类似如下

   ```txt
   sgl_kernel-0.1.4.data/
     └── purelib/
         ├── sgl_kernel/          # 来自 wheel.packages
         │   ├── __init__.py
         │   ├── common_ops.so    # 从临时目录合并而来
         │   └── other_module.py
         └── deep_gemm/          # 来自 install(DIRECTORY)
             ├── __init__.py
             └── include/
                 ├── cute/
                 └── cutlass/
   ```

## torch binding

直接参考由 DeepSeek 总结的步骤

```c++
#include <torch/library.h>       // 必须包含的核心注册头文件
#include <ATen/core/dispatch/Dispatcher.h>  // 调度器（通常隐式包含）
#include "your_custom_kernels.h" // 自定义算子实现的头文件

// 使用 TORCH_LIBRARY_FRAGMENT 注册算子（推荐片段式注册）
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {  // 'sgl_kernel' 是库名
    
    // 步骤 1: 声明算子签名（Schema）
    // 无返回值算子
    m.def("dispose() -> ()");  // 语法：算子名(参数类型) -> 返回值类型
    m.def("meta_size() -> int");
    
    // 含参数算子（Tensor! 表示输出/原位修改）
    m.def("all_reduce(int fa, Tensor inp, Tensor! out, int reg, int reg_sz) -> ()");
    
    // 复杂类型示例（int[] 表示整型列表）
    m.def("init_custom_ar(int[] ipc_tensors, Tensor rank, int rank, bool full_nvlink) -> int");
    
    // 步骤 2: 绑定实现到设备后端
    // CUDA 实现
    m.impl("dispose", torch::kCUDA, &dispose);  // & 指向 C++ 函数
    m.impl("all_reduce", torch::kCUDA, &all_reduce);
    m.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);
    
    // 可选：注册 CPU 实现
    // m.impl("dispose", torch::kCPU, &cpu_dispose);
}
```

核心步骤

1. **算子签名规范**：
   - **输入张量**：用 `Tensor` 声明（如 `Tensor inp`）
   - **输出/原位张量**：用 `Tensor!` 声明（如 `Tensor! out`）
   - **基础类型**：直接写类型名（如 `int`, `bool`, `float`）
   - **复合类型**：用 `类型[]` 表示列表（如 `int[]`）
2. **设备分发**：
   - `torch::kCUDA`/`torch::kCPU` 指定设备后端
   - 同一算子可绑定多设备实现（如 CUDA + CPU）
3. **命名空间**：
   - 库名 `sgl_kernel` 需与 PyTorch 侧调用一致（Python 中 `torch.ops.sgl_kernel.xxx`）
4. **函数实现要求**：
   - 函数签名需严格匹配声明的 Schema（参数顺序/类型）

## A general way to build python extension

1. **定义项目和依赖**
    在 `CMakeLists.txt` 文件中：

   - 使用 `project` 命令定义项目名称和支持的语言（通常包括 C++ 和 CUDA）。

   - 使用 `find_package` 命令查找 Python 和 PyTorch 的库和头文件

2. **设置编译选项**
    根据需要为 C++ 和 CUDA 设置编译选项，例如：

   - 使用 set 命令指定 C++ 标准

   - 使用 `target_compile_options` 设置优化级别或调试选项，确保与 PyTorch 的兼容性。

3. **指定源文件**
    定义一个变量：如 `set(SOURCES file1.cpp file2.cu)`，列出所有需要编译的源文件，通常包括 C++ 文件（.cpp）和 CUDA 文件（.cu）

4. **构建动态链接库**
    使用 `Python_add_library` 命令构建一个 MODULE 类型的动态链接库

5. **链接库**
    使用 `target_link_libraries` 命令将生成的库链接到 Python 和 PyTorch 的库
6. **安装或导出**
    使用 install 命令指定库的安装路径

## Question

- gemm 当中的 ABC layout 分别是 row col row，但是我们传入的 B 矩阵的 layout 应该都是 row，这在 cutlass 当中是怎么进行处理的？

  在 torch 中 transpose 目前是只改变 shape & stride，而不改变内存数据排布

  ```python
  import torch
  
  a = torch.randn((3, 4), device='cuda')
  b = a.t()
  b.is_contiguous()	# False
  b.shape				# (4, 3)
  ```

  在进行 cutlass gemm 的时候通常要求 B matrix 为 column major，以 linear weight 为例，在 pytorch 当中形状为 (N, K)，此时排列为 row major。按照 cutlass 的思想，我们需要的 B matrix 为 (K, N)，排列为 column major。为了完成 cutlass 的要求，其实该 linear weight 只需要重新排列其 shape & stride，内存排布其实是不需要改变的

  ```python
  Cutlass_Layout = (Shape=(K,  N), Stride=(1, K))
  Pytorch_Layout = (Shape=(N,  K), Stride=(K, 1))
  ```

  所以我们在 linear 中使用 cutlass gemm 的时候直接将 weight 其进行 transpose，下面就是 sglang/vllm 在构建 w8a8 linear weight 时的代码，其将 `input_dim & output_dim` 进行了重排

  ```python
  weight = ModelWeightParameter(
      data=torch.empty(
          sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8
      ),
      input_dim=1,
      output_dim=0,
      weight_loader=weight_loader,
  )
  layer.register_parameter("weight", weight)
  ```

- How to do benchmark?

  可以使用 `triton.testing.do_bench` 做一个简单的 benchmark，现在 torch 都是自带 `triton`，用起来比较方便

- 为什么 ccache 不起作用？我似乎还是从头编译的？

- deepgemm 当中的 fp8 是怎么计算的？

  参考 `tests/test_core.py` 计算方式，可以看下分布情况
